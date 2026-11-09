# HexaDepth frontend integration guide

Run the backend:

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload
```

Local backend URL: `http://localhost:8000`

**Inspect the live API schema:** [`/docs`](http://localhost:8000/docs), or `http://localhost:8000/openapi.json`.

## API surface

| Method | Endpoint | Use |
| --- | --- | --- |
| `GET` | `/api/samples` | List available demo sonar scenes |
| `GET` | `/api/samples/{name}` | Load a sample image |
| `POST` | `/api/priority` | Build the progressive transfer and comparison |
| `POST` | `/api/lossless` | Packetize/reassemble full original image with SHA-256 verification |
| `POST` | `/api/analyze` | Analysis and detected-region data |
| `POST` | `/api/preview` | Low-resolution preview |
| `POST` | `/api/link/simulate` | Model link airtime |
| `GET` | `/api/results` | Existing evaluation results |

`POST /api/priority` accepts `multipart/form-data` with either:
- `sample`: a sample name from `/api/samples`, **or**
- `file`: an image upload;
- `rate_bps` (optional): simulated modem bit rate, e.g. `2400`;
- `corrupt_stage` (optional): exercise the simulated corrupt-packet path.

## Browser example

```javascript
const samples = await fetch("http://localhost:8000/api/samples").then(r => r.json());
console.log(samples);

// Use a `sample` name from the samples response, or upload an image.
// The backend demo JS is authoritative for selecting the correct sample key.
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("rate_bps", "2400");

const response = await fetch("http://localhost:8000/api/priority", {
  method: "POST",
  body: form,
});

if (!response.ok) throw new Error(await response.text());
const result = await response.json();
console.log(result.comparison, result.stages);
```

**For your teammate:** The backend serves its own web frontend on port 8000. If you build an independent React/Vite frontend on another port, configure CORS (or a development proxy) before making cross-origin requests. No production authentication, rate limiting, or deployment settings are included.

## What to display honestly

The UI can compare:
1. Time until the first low-resolution overview is available (modeled).
2. Time until prioritized detail crops arrive (modeled).
3. Time to deliver the whole original file exactly (modeled).
4. Relative byte sizes, frame counts, checksum status and reconstruction images.

Do not label simulated airtime as hardware-measured throughput. Do not claim the default ROI selection has an integrated trained model unless actual model weights are installed.

## Tests

`python -m pytest -q` — verified **324 passed, 4 skipped** when prepared.
