# Samvya usage guide

Follow the installation and model setup in [README.md](README.md).

1. Start `python app.py` and open http://127.0.0.1:5002.
2. Create a demonstration profile or select a returning profile.
3. Allow camera access and complete the check-in. This requires the trained emotion checkpoint.
4. Talk with the counsellor using text, or microphone input when Sarvam is configured. Replies use Groq and optional ElevenLabs audio.
5. Choose **Finish & Report** to summarise and save the session.
6. Open **Session History** to view earlier reports. Use the report page's print control and the browser's Save as PDF option for a PDF copy.
7. Developer mode exposes contextual state and simulated wearable data for testing. Simulated values are not hardware measurements.

The local SQLite database stores profiles and transcripts. Keep this prototype on your own computer: reports and profiles currently have no access control. Use demonstration data for testing. Experimental scores and emotion labels are not diagnoses.
