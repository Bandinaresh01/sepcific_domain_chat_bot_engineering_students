# Import required libraries
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API using environment variable
# Ensure GEMINI_API_KEY is set in .env (get from https://makersuite.google.com/)
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Dictionary to store preloaded FAISS vector databases for each subject
subject_vector_dbs = {}

# List of supported subjects
SUBJECTS = ["big_data", "ccv", "krr", "pr"]

# Subject descriptions for prompt refinement and direct answers
SUBJECT_DESCRIPTIONS = {
    "big_data": "Large-scale data processing, Hadoop, Spark, and distributed systems.",
    "ccv": "Image processing, object detection, neural networks, and visual recognition.",
    "krr": "Logical formalisms, ontologies, and reasoning about knowledge in AI systems.",
    "pr": "Classification, clustering, feature extraction, and machine learning patterns, with applications in analyzing text, images, and sounds for fields like Public Relations."
}

def preload_faiss_indexes():
    """
    Preload FAISS indexes for all subjects at application startup.
    Populates subject_vector_dbs with loaded vector stores.
    """
    print("⏳ Preloading FAISS indexes for all subjects...")
    for subject in SUBJECTS:
        vector_db = load_faiss_index(subject)
        if vector_db:
            subject_vector_dbs[subject] = vector_db
        else:
            print(f"❌ Failed to preload FAISS index for {subject}")
    print("✅ All FAISS indexes preloaded.")

def load_faiss_index(subject_name):
    """
    Load the FAISS index for a given subject.
    Args:
        subject_name (str): The subject code (e.g., 'pr').
    Returns:
        FAISS vector store or None if loading fails.
    """
    index_path = f"faiss_index/{subject_name}_faiss"
    print(f"Checking FAISS index at: {index_path}")
    
    if not os.path.exists(index_path):
        print(f"❌ No FAISS index found for {subject_name}")
        return None
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        # Load the FAISS index
        vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        print(f"✅ Loaded FAISS index for {subject_name}")
        return vector_db
    except Exception as e:
        print(f"❌ Error loading FAISS index: {str(e)}")
        return None

def refine_query(user_query, current_subject):
    """
    Refine the user's query to be subject-specific using Gemini, with PR context for 'pr' subject.
    Args:
        user_query (str): The user's raw question.
        current_subject (str): The subject code (e.g., 'pr').
    Returns:
        str: Refined query tailored to the subject.
    """
    try:
        subject_description = SUBJECT_DESCRIPTIONS.get(current_subject, "General subject knowledge")
        context = (
            "For Public Relations, focus on applications of machine learning techniques like classification, clustering, and feature extraction in analyzing text, images, or sounds."
            if current_subject == "pr" else ""
        )
        prompt = f"""
        You are an expert in refining questions to be specific to a subject.
        Subject: {current_subject}
        Subject Description: {subject_description}
        {context}
        User Question: {user_query}
        
        Rewrite the question to be precise and relevant to the subject, ensuring it aligns with the subject's syllabus content.
        For Public Relations ('pr'), emphasize machine learning applications in PR contexts like sentiment analysis or media monitoring.
        If the question is already clear, return it unchanged.
        Return only the refined question as a single sentence.
        """
        print(f"Refining query: {user_query}")
        response = gemini_model.generate_content(prompt)
        refined_query = response.text.strip()
        print(f"Refined query: {refined_query}")
        return refined_query
    except Exception as e:
        print(f"❌ Error refining query: {str(e)}")
        return user_query  # Fallback to original query if refinement fails

def get_answer(user_query, current_subject, session_histories):
    """
    Process a user query and return an answer based on the syllabus or Gemini.
    Args:
        user_query (str): The user's question.
        current_subject (str): The subject code (e.g., 'pr').
        session_histories (dict): Dictionary storing conversation history.
    Returns:
        str: The answer from FAISS or Gemini, or an error message.
    """
    try:
        print(f"Processing query: {user_query} for subject: {current_subject}")
        
        # Refine the query to be subject-specific
        refined_query = refine_query(user_query, current_subject)

        # Check if the subject's FAISS index is preloaded
        vector_db = subject_vector_dbs.get(current_subject)
        if not vector_db:
            print(f"❌ No FAISS index available for {current_subject}")
            return generate_direct_answer(user_query, refined_query, current_subject, session_histories)

        # Get previous conversation history for the subject
        previous_chats = session_histories.get(current_subject, [])
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in previous_chats])
        print(f"Conversation history length: {len(history_text)} characters")

        # Retrieve relevant documents from FAISS
        print("Retrieving documents from FAISS...")
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(refined_query)
        print(f"Retrieved {len(docs)} documents")
        
        # Combine document content
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"Combined text length: {len(combined_text)} characters")

        # If no documents are found, fallback to Gemini
        if not docs:
            print(f"❌ No relevant documents found for {refined_query}")
            return generate_direct_answer(user_query, refined_query, current_subject, session_histories)

        # Create the prompt for Gemini using FAISS documents
        context = (
            "For Public Relations, focus on applications of machine learning techniques like classification, clustering, and feature extraction in PR contexts such as sentiment analysis, media monitoring, or audience segmentation."
            if current_subject == "pr" else ""
        )
        prompt = f"""
        You are a helpful assistant specializing in {current_subject}.
        
        Conversation History:
        {history_text}
        
        Answer the following question based ONLY on the provided PDF contents.
        {context}
        If the answer is not explicitly available, use the context to provide a concise, relevant response, focusing on machine learning applications for Public Relations if the subject is 'pr'.
        
        PDF Contents:
        {combined_text}
        
        User Question:
        {refined_query}
        
        Answer:
        """
        print(f"Prompt length: {len(prompt)} characters")

        # Generate answer using Gemini
        print("Sending prompt to Gemini API...")
        response = gemini_model.generate_content(prompt)
        answer_text = response.text
        print(f"Received response: {answer_text[:100]}...")

        # Update session history with original user query
        if current_subject not in session_histories:
            session_histories[current_subject] = []
        session_histories[current_subject].append((user_query, answer_text))
        print(f"Updated session history for {current_subject}")

        return answer_text

    except Exception as e:
        print(f"❌ Error answering question: {str(e)}")
        return generate_direct_answer(user_query, refined_query, current_subject, session_histories)

def generate_direct_answer(original_query, refined_query, current_subject, session_histories):
    """
    Generate an answer directly using Gemini when FAISS fails or no documents are found.
    Args:
        original_query (str): The original user question.
        refined_query (str): The refined subject-specific question.
        current_subject (str): The subject code.
        session_histories (dict): Dictionary storing conversation history.
    Returns:
        str: Answer generated by Gemini.
    """
    try:
        subject_description = SUBJECT_DESCRIPTIONS.get(current_subject, "General subject knowledge")
        previous_chats = session_histories.get(current_subject, [])
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in previous_chats])

        context = (
            "For Public Relations, provide detailed examples of machine learning techniques like classification (e.g., sentiment analysis), clustering (e.g., audience segmentation), and feature extraction (e.g., text feature analysis) applied to PR tasks such as media monitoring, crisis management, or campaign optimization."
            if current_subject == "pr" else ""
        )
        prompt = f"""
        You are an expert in {current_subject}.
        Subject Description: {subject_description}
        
        Conversation History:
        {history_text}
        
        The syllabus content is unavailable or insufficient to answer the question.
        {context}
        Provide a concise and accurate answer to the following question based on your knowledge of {current_subject}.
        Ensure the answer is relevant to the subject and suitable for a syllabus-based context, with specific examples for Public Relations if the subject is 'pr'.
        
        Question:
        {refined_query}
        
        Answer:
        """
        print(f"Generating direct answer for: {refined_query}")
        response = gemini_model.generate_content(prompt)
        answer_text = response.text
        print(f"Direct answer: {answer_text[:100]}...")

        # Update session history with original user query
        if current_subject not in session_histories:
            session_histories[current_subject] = []
        session_histories[current_subject].append((original_query, answer_text))
        print(f"Updated session history for {current_subject} (direct answer)")

        return answer_text
    except Exception as e:
        print(f"❌ Error generating direct answer: {str(e)}")
        return "❌ Unable to generate an answer at this time. Please try again."