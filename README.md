# Agricultural Skills Recommendation Chatbot

## Setup Instructions

Follow the steps below to run the application locally.

---

## 1. Clone the Repository


---

## 2. Access the Project Directory

Open Command Prompt (CMD) and navigate to the project folder.


---

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate the Virtual Environment

```bash
venv\scripts\activate
```

---

## 5. Install the Required Dependencies

Install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 6. Configure the Secret Keys

Rename:

```text
.streamlit/secrets.toml.example
```

to:

```text
.streamlit/secrets.toml
```

Then add the required secret keys and credentials inside the file.

Example:

```toml
NEO4J_URI="your_neo4j_uri"
NEO4J_USERNAME="your_username"
NEO4J_PASSWORD="your_password"
GOOGLE_API_KEY="your_openai_api_key"
```

---

## 7. Run the Application

```bash
streamlit run app.py
```

---

## Notes

* Make sure Python is installed before starting the setup.
* It is recommended to use Python 3.10 or newer.
* Do not upload the `secrets.toml` file to GitHub.
