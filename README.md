# KTUGPT-Python

A Flask web application that is designed for answering questions based on the context from the PDFs. It uses the [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model as the large language model (LLM) and the [hkunlp/instructor-xl](https://huggingface.co/hkunlp/instructor-xl) model for embedding text representations.


## Setup

- Clone this repository:

   ```
   git clone https://github.com/sameemul-haque/ktugpt-python.git
   ```
- After cloning the repository, navigate into the ktugpt-python directory

   ```
   cd ktugpt-python
   ```

- Set up a Python virtual environment:

   ```
   python -m venv venv
   ```

- Activate the virtual environment:

   - GNU/Linux | MacOS:
     ```
     source venv/bin/activate
     ```
   - Windows:
     ```
     venv\Scripts\activate
     ```

- Install dependencies:

   ```
   pip install -r requirements.txt
   ```

6. Create a `.env` file based on `.env.example` and add your [Hugging Face API token](https://huggingface.co/docs/hub/en/security-tokens).

- Run the Flask web app:

   ```
   flask run --app app --host=0.0.0.0
   ```

## Usage

Once the Flask app is running, you can send GET requests to `http://127.0.0.1:5000` with a query parameter `q` containing your question. The app will return an answer based on the configured language model and retrieval method. For example, `http://127.0.0.1:5000/?q=what%20is%20operating%20system?`

![preview](https://raw.githubusercontent.com/sameemul-haque/KTUGPT-Python/preview/preview.png "preview")

