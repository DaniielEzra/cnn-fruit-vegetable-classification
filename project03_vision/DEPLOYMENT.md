# Bonus: phone testing, ngrok and cloud deployment

None of these have been set up for you. Follow the steps yourself and only report them in your submission after they work.

Before you start, the website must already work on your computer at http://127.0.0.1:8000 (see README.md).

---

## A. Test from your phone with ngrok (recommended)

ngrok gives your local server a temporary public **https** address. The phone does not need to be on the same Wi-Fi.

1. Create a free account at [ngrok.com](https://ngrok.com) and sign in.
2. Download ngrok for Windows from your ngrok dashboard and unzip it (or install it with `winget install ngrok.ngrok`).
3. Copy your **authtoken** from the dashboard (**Getting Started > Your Authtoken**) and run once:

   ```powershell
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```

   Never put the token in GitHub.

4. Terminal 1, start the website:

   ```powershell
   .venv\Scripts\activate
   uvicorn app:app --port 8000
   ```

5. Terminal 2, start the tunnel:

   ```powershell
   ngrok http 8000
   ```

6. ngrok shows a line like `Forwarding https://something.ngrok-free.app -> http://localhost:8000`. Open that **https** address on your phone.
7. The free plan may show an ngrok warning page first; tap **Visit Site**.
8. Tap **Take Photo** (opens the camera on most phones) or **Choose Image**, then **Classify Image**.
9. Take a phone screenshot that shows the prediction and save it as `screenshots/phone_prediction.png`.

Notes:
* Anyone with the link can use your app while ngrok runs. Stop it with **Ctrl+C** when you are done.
* The free address changes each time you restart ngrok.
* Your computer must stay on with both terminals running.

---

## B. Test from your phone on the same Wi-Fi (no ngrok)

1. Find your computer's IP address: run `ipconfig` and look for **IPv4 Address**, for example `192.168.1.25`.
2. Start the server so other devices can connect:

   ```powershell
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. If Windows Firewall asks, allow access on **Private networks**.
4. On the phone (same Wi-Fi), open `http://192.168.1.25:8000` (use your own IP).

Upload and the **Take Photo** button usually work over plain http because they use the phone's normal file picker. If your browser refuses, use the ngrok method, which gives https.

---

## C. Optional: deploy to a public cloud service

A cloud deployment keeps the site online when your computer is off. Free plans change often, so check the current limits. TensorFlow needs roughly 1 to 2 GB of memory, which is more than some free plans allow.

### Option 1: Hugging Face Spaces (Docker)

1. Create an account at [huggingface.co](https://huggingface.co), then **New Space**, SDK **Docker**, template **Blank**.
2. Add this file as `Dockerfile` in the project (replace `X.Y.Z` with the `tensorflow_version` from `model/model_info.json`):

   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir tensorflow-cpu==X.Y.Z
   COPY . .
   EXPOSE 7860
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

3. Upload the project files to the Space (including `model/vision_model.keras`, `model/model_info.json`, `class_names.json`, `static/`). Large binary files may need Git LFS, which Spaces supports.
4. Wait for the build. Open the Space URL and test `/health`.

### Option 2: Render (render.com)

1. Push the project to GitHub (README section 9).
2. On Render: **New > Web Service**, connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. If the service runs out of memory, choose a plan with more RAM.

After deploying, open `https://YOUR-URL/health`: it should return `"model_loaded": true`. Put the public URL in your README only after it works.
