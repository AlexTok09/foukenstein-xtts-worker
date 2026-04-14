# runpod_handler — Worker Serverless XTTS

Image Docker + handler Python pour l'endpoint Runpod Serverless qui sert
la synthèse vocale Foukenstein (XTTS v2 fine-tuné).

```
runpod_handler/
├── handler.py          ← entrypoint Runpod (port de xtts_http_server.py)
├── requirements.txt    ← TTS 0.22.0, runpod, soundfile, numpy<2
├── Dockerfile          ← base cuda 11.8 + python 3.10 + torch 2.1.0 cu118
├── .dockerignore
└── README.md           ← ce fichier
```

Les **poids** XTTS ne sont PAS dans l'image — ils sont lus depuis le Network
Volume `srvvzpk6jj` monté à `/runpod-volume/`. Voir
`deployment_v1/NETWORK_VOLUME.md` pour la cartographie complète.

---

## Contrat I/O

**Input** (reçu via `event["input"]`) :
```json
{
  "chunks":     ["...", "..."],
  "language":   "fr",
  "out_format": "wav"
}
```

**Output** (renvoyé à Runpod, serialisé dans `output`) :
```json
{
  "audio_base64": "UklGR...",
  "format":       "wav",
  "chunks":       5,
  "duration_ms":  12400,
  "elapsed_ms":   3800
}
```

Ce contrat correspond exactement à ce qu'attend
`deployment_v1/app/runpod_tts_client.py`.

---

## Build de l'image

**Ce pod Runpod n'a pas de daemon Docker.** Build l'image sur une machine
avec Docker (laptop, CI, ou autre pod avec Docker-in-Docker).

```bash
cd deployment_v1/runpod_handler/

# Build
docker build -t tonuser/foukenstein-xtts:0.1 .

# Test local rapide (sans GPU — vérifie juste que l'image boot et que les
# deps s'importent ; le model load échouera sans /runpod-volume mais ça
# permet d'attraper les problèmes de packaging)
docker run --rm tonuser/foukenstein-xtts:0.1 python -c "import TTS, runpod, soundfile; print('imports OK')"

# Push
docker push tonuser/foukenstein-xtts:0.1
```

**Taille attendue** : ~5-6 GB (torch + CUDA libs + TTS). C'est la norme
pour du PyTorch GPU, rien à optimiser.

> **Astuce build cache** : le layer `torch` est séparé du layer `requirements.txt`
> exprès — modifier les deps Python ne re-télécharge pas les 2 GB de torch.

---

## Création de l'endpoint Runpod

Sur la console Runpod → **Serverless → New Endpoint → Custom Source** :

| Champ | Valeur |
|---|---|
| **Container Image** | `tonuser/foukenstein-xtts:0.1` |
| **GPU Type** | A4000 ou A5000 (24 GB VRAM — XTTS fp32 tient large) |
| **Region** | **EU-RO-1** (obligatoire — le volume est region-locked) |
| **Network Volume** | `srvvzpk6jj` monté sur `/runpod-volume/` |
| **Min workers** | `0` (scale-to-zero, accepter cold start ~10-20 s avec FlashBoot) |
| **Max workers** | `1` pour démarrer, `2-3` si charge |
| **Idle timeout** | `300` s (garde le worker chaud 5 min entre requêtes) |
| **Execution timeout** | `180` s |
| **FlashBoot** | activé |

### Env vars à définir sur l'endpoint

```
FTCKPT=/runpod-volume/xtts/best_model_19875.pth
ORIG=/runpod-volume/xtts/XTTS_v2.0_original_model_files
SPEAKERS_PTH=/runpod-volume/xtts/XTTS_v2.0_original_model_files/speakers_xtts.pth
SPEAKER_WAV=/runpod-volume/xtts/test_speaker.wav
LANG=fr
```

Les params de stitching ont des défauts figés dans `handler.py` (identiques
à `start_tts_frozen.sh`), pas besoin de les redéfinir sauf override volontaire :
`XFADE_MS=320`, `EDGE_FADE_MS=80`, `PAUSE_MS=160`, `MICRO_PAUSE_MS=40`,
`TAIL_SILENCE_MS=700`.

`SPEAKERS_PTH` a un défaut dans le handler (dérivé de `ORIG`) — tu peux
l'omettre, mais le mettre explicitement rend la config plus lisible.

---

## Test bout-en-bout

Une fois l'endpoint créé, récupère son `ENDPOINT_ID` dans l'URL de la
console Runpod, puis :

```bash
export RUNPOD_API_KEY="..."
export ENDPOINT_ID="xxxxxxxxxxxx"

curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"chunks":["Bonjour, ceci est un test.","La seconde phrase valide le stitching."],"language":"fr","out_format":"wav"}}' \
  | python -c 'import sys,json,base64; d=json.load(sys.stdin); open("/tmp/test.wav","wb").write(base64.b64decode(d["output"]["audio_base64"])); print("OK", d["output"])'

# Ecoute : aplay /tmp/test.wav  (ou transfère sur une machine avec haut-parleur)
```

Premier appel = cold start (~10-20 s avec FlashBoot, plus long au tout premier
boot avant le snapshot). Appels suivants = warm, ~1-3 s de latence TTS
côté serveur + réseau.

Si le test direct passe, renseigne `RUNPOD_SERVERLESS_ENDPOINT_ID` dans
`deployment_v1/env.sh` et relance le web server : `POST /api/ask` doit
maintenant produire un WAV via `/audio/<uuid>.wav`.

---

## Debug

Logs du worker : **console Runpod → Serverless → [ton endpoint] → Logs**.

Erreurs probables au premier boot :
- `FileNotFoundError: /runpod-volume/xtts/...` → le Network Volume n'est pas
  attaché à l'endpoint, ou mauvaise région. Vérifier la config endpoint.
- `CUDA out of memory` → très improbable sur A4000 (24 GB), mais si ça
  arrive c'est qu'un autre process squatte la VRAM. Redémarrer le worker.
- `ModuleNotFoundError: TTS` → image mal buildée, re-push.
- Retourne `{"error": "..."}` dans `output` → erreur attrapée côté handler,
  lire le traceback dans les logs Runpod.
