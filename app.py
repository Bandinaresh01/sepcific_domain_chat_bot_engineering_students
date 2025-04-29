# Import required libraries
import os
from flask import Flask, render_template, request
import query

# Initialize Flask app, specifying the templates folder
app = Flask(__name__, template_folder='templates')

# Global dictionary to store conversation history for each subject
session_histories = {}

# Preload FAISS indexes at application startup
query.preload_faiss_indexes()

@app.route('/')
def index():
    """
    Display the subject selection page.
    Returns: Rendered index.html template with list of subjects.
    """
    subjects = ["big_data", "ccv", "krr", "pr"]
    return render_template("index.html", subjects=subjects)

@app.route('/ask/<subject>', methods=['GET', 'POST'])
def ask(subject):
    """
    Handle the question-asking page for a specific subject.
    Args:
        subject (str): The subject code (e.g., 'pr').
    Returns: Rendered ask.html template with chat interface and responses.
    """
    global session_histories
    print(f"Accessing subject: {subject}, History: {session_histories.get(subject, [])}")

    if request.method == 'POST':
        user_query = request.form['user_query']
        print(f"Received query: {user_query}")
        answer = query.get_answer(user_query, subject, session_histories)
        print(f"Generated answer: {answer[:100]}...")
        return render_template("ask.html", 
                             subject=subject, 
                             question=user_query, 
                             answer=answer, 
                             session_histories=session_histories)
    
    return render_template("ask.html", 
                         subject=subject, 
                         session_histories=session_histories)

if __name__ == "__main__":
    app.run(debug=True)