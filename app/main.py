import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from typing import Dict, List, Optional, AsyncGenerator
import re
import json
from fastapi.responses import StreamingResponse
import asyncio
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI(
    title="BSSRV RAG Chatbot API",
    description="API for BSSRV University chatbot",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://bssrv.netlify.app", "https://b8e0-2409-40e6-135-89d0-1d03-466d-b8c6-2b43.ngrok-free.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Verify API key
if not GEMINI_API_KEY:
    raise ValueError("API key not found in .env file. Please set it and try again.")

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

class KnowledgeBase:
    def __init__(self, name: str, description: str, vector_store: Optional[Chroma] = None):
        self.name = name
        self.description = description
        self.vector_store = vector_store

# Dictionary to store multiple knowledge bases
knowledge_bases: Dict[str, KnowledgeBase] = {}

def initialize_vector_store(document_paths: List[str], kb_name: str) -> Optional[Chroma]:
    try:
        print(f"Initializing vector store for {kb_name} with comprehensive information...")
        all_docs = []
        
        # Use absolute path with app directory
        doc_path = document_paths[0]  # We're only using one document now
        full_path = os.path.join(os.path.dirname(__file__), doc_path)
        
        if not os.path.exists(full_path):
            print(f"Error: {full_path} not found for knowledge base {kb_name}.")
            return None
            
        print(f"Loading document: {full_path}")
        
        # Load the text file
        loader = TextLoader(full_path)
        documents = loader.load()
        print(f"Loaded text file: {doc_path}")
        
        all_docs.extend(documents)
        
        if not all_docs:
            print(f"Error: Document could not be loaded for knowledge base {kb_name}.")
            return None
            
        print(f"Splitting document for {kb_name}...")
        # Use smaller chunk size with higher overlap for more precise retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        docs = text_splitter.split_documents(all_docs)
        print(f"Created {len(docs)} chunks for {kb_name}")
        
        print(f"Creating embeddings for {kb_name}...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        persist_dir = os.path.join(os.path.dirname(__file__), f"chroma_db_{kb_name}")
        print(f"Creating vector store at {persist_dir}")
        vector_store = Chroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=persist_dir
        )
        print(f"Vector store for {kb_name} created successfully with {len(docs)} chunks")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store for {kb_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def initialize_knowledge_bases():
    # Define a new knowledge base with a more specific name to replace the "general" one
    kb_config = {
        "name": "bssrv_university_information",
        "description": "Comprehensive BSSRV University academic programs and admission information",
        "document_paths": [
            "bssrv_general_info_updated.txt"
        ]
    }

    print(f"Loading knowledge base: {kb_config['name']} with comprehensive university information")
    vector_store = initialize_vector_store(kb_config["document_paths"], kb_config["name"])
    knowledge_bases[kb_config["name"]] = KnowledgeBase(
        name=kb_config["name"],
        description=kb_config["description"],
        vector_store=vector_store
    )
    
    # Verify knowledge base loaded correctly
    if knowledge_bases[kb_config["name"]].vector_store:
        print(f"✅ Knowledge base '{kb_config['name']}' loaded successfully with comprehensive information")
    else:
        print(f"❌ Knowledge base '{kb_config['name']}' failed to load")

# Initialize knowledge bases at startup
print("Starting knowledge base initialization...")
initialize_knowledge_bases()
print("Knowledge base initialization complete")

# Place this near the beginning of the file, after the initial imports and before the FastAPI app initialization
def remove_duplicate_questions(response: str) -> str:
    """
    Post-processes the LLM response to remove any duplicate or multiple questions.
    This ensures the AI only asks one focused question at a time.
    Also cleans up any malformed URLs and removes unprompted questions.
    """
    # First, clean up malformed URLs
    cleaned_response = response
    
    # Fix common URL problems
    # 1. Fix URLs with weird extensions
    cleaned_response = re.sub(r'(https?://[^\s]+)\.(heads|html?|php|aspx?)(?=[\s,.!?]|$)', r'\1', cleaned_response, flags=re.IGNORECASE)
    
    # 2. Fix duplicate URLs (like https://example.com.https://example.com)
    cleaned_response = re.sub(r'(https?://[^\s.]+\.[^\s.]+)\.(https?://)', r'\1 \2', cleaned_response, flags=re.IGNORECASE)
    
    # Common patterns where the AI starts a new question after already asking one
    patterns = [
        r'\?[\s\n]+So you\'re interested in',
        r'\?[\s\n]+Would you like to',
        r'\?[\s\n]+Can you tell me',
        r'\?[\s\n]+Do you have any',
        r'\?[\s\n]+Are you looking',
        r'\?[\s\n]+Have you considered',
        r'\?[\s\n]+What about',
        r'\?[\s\n]+So you\'re',
        r'\?[\s\n]+Any questions about',
        r'\?[\s\n]+You seem really interested',
        r'\?[\s\n]+What would you like to know',
        r'\?[\s\n]+Is there anything else',
        r'\?[\s\n]+Would you like me to',
    ]
    
    # Check for these patterns and cut off the response if found
    for pattern in patterns:
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            # Cut off at the question mark
            return cleaned_response[:match.start() + 1]
    
    # Additional patterns to catch the application form prompt specifically
    app_form_patterns = [
        r'You seem really interested in our B\.Tech program\.',
        r'Have you filled out an application form yet\?',
        r'I can provide you with the link to apply online',
        r'Would you like the link to apply'
    ]
    
    for pattern in app_form_patterns:
        if re.search(pattern, cleaned_response, re.IGNORECASE):
            # Find the last complete sentence before this prompt
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
            for i in range(len(sentences) - 1, -1, -1):
                if re.search(pattern, sentences[i], re.IGNORECASE):
                    return '. '.join(sentences[:i]) + '.'
    
    # If no specific patterns matched but there are multiple question marks,
    # find all question marks in the response
    question_marks = [m.start() for m in re.finditer(r'\?', cleaned_response)]
    
    # If there are multiple question marks, consider cutting at the first question mark after a certain length
    if len(question_marks) > 1:
        # Find the first question mark that's sufficiently into the response
        for position in question_marks:
            if position > 100:  # Only consider cutting if we have a substantial response first
                # Get next few characters to check if we're in the middle of a list
                next_chars = cleaned_response[position+1:position+20] if position+20 < len(cleaned_response) else cleaned_response[position+1:]
                # If the next part doesn't start with list markers or conjunctions, cut
                if not re.match(r'^[\s\n]*(or|and|,|\d\.|\*|\-)', next_chars, re.IGNORECASE):
                    return cleaned_response[:position+1]
    
    return cleaned_response

# Function to detect agriculture-related queries
def is_agriculture_query(query: str) -> bool:
    """Check if a query is asking about agriculture programs"""
    query_lower = query.lower()
    agriculture_terms = [
        'agriculture', 'bsc agriculture', 'agri', 'farming', 'farm science',
        'agricultural', 'bsc agri', 'b.sc agriculture', 'b.sc. agriculture',
        'horticulture', 'crop', 'soil science', 'plant science'
    ]
    return any(term in query_lower for term in agriculture_terms)

# Replace query_gemini with query_gemini
def query_gemini(prompt: str, user_name: Optional[str] = None, kb_name: Optional[str] = None, model_name: str = "gemini-2.0-flash", is_assamese_english_mixed: bool = False):
    # Enhanced system message with strong emphasis on maintaining conversation context and using context data
    system_message = """You are a friendly and helpful BSSRV University assistant. Respond in a natural, conversational way as if you're having a casual chat. 

IMPORTANT GUIDELINES:
1. Never use phrases like 'According to the provided context' or 'Based on the information provided'.
2. Answer user queries based ONLY on the provided context data. ALWAYS USE THE INFORMATION FROM THE CONTEXT!
3. When asked about specific information like links, facilities, admission processes, or contact details, ALWAYS refer to the provided context and EXTRACT the correct information.
5. MAINTAIN CONVERSATION CONTEXT AT ALL TIMES. 
   - You MUST remember what was previously discussed and continue the conversation naturally.
   - If the user responds with short answers like "yes", "yeah", "no", etc., you must continue the conversation based on what was just discussed.
   - Never treat each message as a new, separate conversation.
6. If the user's message is a short affirmative reply (like "yes", "yeah", "sure"):
   - Continue elaborating on the previous topic you were discussing
   - Or provide a direct answer to the question you just asked
   - NEVER start a new, unrelated topic
7. For short negative replies (like "no"), acknowledge their response and offer alternatives related to the current topic.
8. NEVER END YOUR RESPONSE WITH A QUESTION. If you want to ask a question, do it in the middle of your response, but not at the end.
9. DO NOT ask if the user wants more information, has questions, or needs anything else - wait for them to ask.
10. NEVER generate follow-up questions at the end of your response like "Do you have any questions?" or "What would you like to know next?".
11. If the information isn't in your context:
    - For simple questions: Simply acknowledge you don't have that specific information right now.
    - For complex questions: You may briefly suggest contacting the university.
12. Never suggest joining the WhatsApp group as your primary response to any question.
13. REVIEW your previous messages before responding to maintain a coherent, flowing conversation.
14. WHEN PROVIDING URLs:
    - Make sure each URL is correctly formatted
    - Do NOT repeat the same URL multiple times in a row
    - Leave a space between URLs if you need to provide multiple links
    - DO NOT attach invalid extensions to URLs (like .ant, .html, .heads when they don't exist)
15. DO NOT add an application form prompt at the end of your message like "Have you filled out an application form yet?" or "You seem really interested in our program" - this will be added separately if needed.
16. ALWAYS PRIORITIZE INFORMATION FROM THE CONTEXT OVER SAYING "I DON'T HAVE THAT INFORMATION".

CRITICAL INSTRUCTION: BSSRV University DOES NOT offer ANY agriculture programs, BSc Agriculture, or agricultural sciences. If asked about agriculture programs, clearly state that BSSRV does not offer such programs and only offers B.Tech programs. DO NOT provide any information about how to apply for agricultural programs, as they DO NOT EXIST at BSSRV University. There is NO agriculture department, NO agriculture website, and NO agriculture admission process.
"""

    # Add language instructions for Assamese-English mixed language
    if is_assamese_english_mixed:
        system_message += """
LANGUAGE STYLE REQUIREMENT - EXTREMELY IMPORTANT: 
1. The user is writing in a mix of English and Assamese (Anglo-Assamese written in English script).
2. You MUST respond in the SAME mixed language style - use Assamese words written in English script mixed with English.
3. NEVER use Bengali script (অ, আ, ক, etc.) or any other non-Latin script in your response.
4. Common Assamese words to use: moi (I), amar/mor (my), apuni (you formal), tumi (you informal), ase (is/have), nai (is not/don't have), kiman (how much), kene (how), kisu (something), koribo (will do), korim (I will do), lage (need/require).
5. Keep technical terms, program names, and college names in English.
6. Example format: "Apuni 60% paisile, so apuni eligible ase. CSE program pabo ba napabo depends koribo entrance exam or marks upot."
7. This is CRITICAL: NEVER use Bengali script characters. Only use Latin/English alphabet to write Assamese words.
"""

    if kb_name and kb_name in knowledge_bases:
        if kb_name == "general":
            system_message += " You are currently using the general university information knowledge base. You should provide helpful information about BSSRV University, its history, campus, facilities, departments, and courses based on the context provided."
        
        # Only include user-specific information when available
        if user_name:
            system_message += f" The user's name is {user_name}. Please address them by their name when appropriate."

    try:
        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}"
        
        # Create the Gemini model and generate content
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(full_prompt)
        
        response_text = response.text
        
        # If Assamese-English mixed mode and response contains Bengali script, regenerate with stronger instructions
        if is_assamese_english_mixed and any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in response_text):
            # This detects Bengali Unicode range
            print("Detected Bengali script in response when Assamese-English mixed was requested. Regenerating...")
            stronger_prompt = full_prompt + "\n\nCRITICAL ERROR: Your previous response contained Bengali script characters. NEVER use Bengali script. Respond ONLY using English alphabet/Latin characters even for Assamese words."
            response = model.generate_content(stronger_prompt)
            response_text = response.text
            
        # Apply post-processing to remove duplicate questions
        response_text = remove_duplicate_questions(response_text)
        
        return response_text
    except Exception as e:
        print(f"Error in query_gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying Gemini API: {str(e)}")

# Update the streaming function to use Gemini instead of Groq
async def query_gemini_stream(prompt: str, user_name: Optional[str] = None, kb_name: Optional[str] = None, is_assamese_english_mixed: bool = False) -> AsyncGenerator[str, None]:
    # Enhanced system message with strong emphasis on maintaining conversation context and using context data
    system_message = """You are a friendly and helpful BSSRV University assistant. Respond in a natural, conversational way as if you're having a casual chat. 

IMPORTANT GUIDELINES:
1. Never use phrases like 'According to the provided context' or 'Based on the information provided'.
2. Answer user queries based ONLY on the provided context data. ALWAYS USE THE INFORMATION FROM THE CONTEXT!
3. When asked about specific information like links, facilities, admission processes, or contact details, ALWAYS refer to the provided context and EXTRACT the correct information.
5. MAINTAIN CONVERSATION CONTEXT AT ALL TIMES. 
   - You MUST remember what was previously discussed and continue the conversation naturally.
   - If the user responds with short answers like "yes", "yeah", "no", etc., you must continue the conversation based on what was just discussed.
   - Never treat each message as a new, separate conversation.
6. If the user's message is a short affirmative reply (like "yes", "yeah", "sure"):
   - Continue elaborating on the previous topic you were discussing
   - Or provide a direct answer to the question you just asked
   - NEVER start a new, unrelated topic
7. For short negative replies (like "no"), acknowledge their response and offer alternatives related to the current topic.
8. NEVER END YOUR RESPONSE WITH A QUESTION. If you want to ask a question, do it in the middle of your response, but not at the end.
9. DO NOT ask if the user wants more information, has questions, or needs anything else - wait for them to ask.
10. NEVER generate follow-up questions at the end of your response like "Do you have any questions?" or "What would you like to know next?".
11. If the information isn't in your context:
    - For simple questions: Simply acknowledge you don't have that specific information right now.
    - For complex questions: You may briefly suggest contacting the university.
12. Never suggest joining the WhatsApp group as your primary response to any question.
13. REVIEW your previous messages before responding to maintain a coherent, flowing conversation.
14. WHEN PROVIDING URLs:
    - Make sure each URL is correctly formatted
    - Do NOT repeat the same URL multiple times in a row
    - Leave a space between URLs if you need to provide multiple links
    - DO NOT attach invalid extensions to URLs (like .ant, .html, .heads when they don't exist)
15. DO NOT add an application form prompt at the end of your message like "Have you filled out an application form yet?" or "You seem really interested in our program" - this will be added separately if needed.
16. ALWAYS PRIORITIZE INFORMATION FROM THE CONTEXT OVER SAYING "I DON'T HAVE THAT INFORMATION".

CRITICAL INSTRUCTION: BSSRV University DOES NOT offer ANY agriculture programs, BSc Agriculture, or agricultural sciences. If asked about agriculture programs, clearly state that BSSRV does not offer such programs and only offers B.Tech programs. DO NOT provide any information about how to apply for agricultural programs, as they DO NOT EXIST at BSSRV University. There is NO agriculture department, NO agriculture website, and NO agriculture admission process.
"""

    # Add language instructions for Assamese-English mixed language
    if is_assamese_english_mixed:
        system_message += """
LANGUAGE STYLE REQUIREMENT - EXTREMELY IMPORTANT: 
1. The user is writing in a mix of English and Assamese (Anglo-Assamese written in English script).
2. You MUST respond in the SAME mixed language style - use Assamese words written in English script mixed with English.
3. NEVER use Bengali script (অ, আ, ক, etc.) or any other non-Latin script in your response.
4. Common Assamese words to use: moi (I), amar/mor (my), apuni (you formal), tumi (you informal), ase (is/have), nai (is not/don't have), kiman (how much), kene (how), kisu (something), koribo (will do), korim (I will do), lage (need/require).
5. Keep technical terms, program names, and college names in English.
6. Example format: "Apuni 60% paisile, so apuni eligible ase. CSE program pabo ba napabo depends koribo entrance exam or marks upot."
7. This is CRITICAL: NEVER use Bengali script characters. Only use Latin/English alphabet to write Assamese words.
"""

    if kb_name and kb_name in knowledge_bases:
        if kb_name == "general":
            system_message += " You are currently using the general university information knowledge base. You should provide helpful information about BSSRV University, its history, campus, facilities, departments, and courses based on the context provided."
            
        # Only include user-specific information when available
        if user_name:
            system_message += f" The user's name is {user_name}. Please address them by their name when appropriate."

    try:
        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}"
        
        # Generate response from Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        
        response_text = response.text
        
        # Check if response contains Bengali script characters when Assamese-English was requested
        if is_assamese_english_mixed and any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in response_text):
            # This detects Bengali Unicode range
            print("Detected Bengali script in streaming response when Assamese-English mixed was requested. Regenerating...")
            stronger_prompt = full_prompt + "\n\nCRITICAL ERROR: Your previous response contained Bengali script characters. NEVER use Bengali script. Respond ONLY using English alphabet/Latin characters even for Assamese words."
            response = model.generate_content(stronger_prompt)
            response_text = response.text
        
        # For streaming simulation, we'll break the response into chunks
        chunk_size = 15  # Characters per chunk
        
        all_content = ""
        
        # Simulate streaming by yielding the response in small chunks
        for i in range(0, len(response_text), chunk_size):
            content = response_text[i:i+chunk_size]
            all_content += content
            # Small delay to simulate streaming
            await asyncio.sleep(0.05)
            yield content
            
        # Apply post-processing to the complete response
        try:
            if all_content:
                processed_response = remove_duplicate_questions(all_content)
                
                # If the post-processed response is different from the original, 
                # we need to inform the client somehow, but we can't change the stream after it's sent
                if processed_response != all_content:
                    print("Response was post-processed, but changes can't be streamed")
        except Exception as e:
            print(f"Error in post-processing: {e}")
            
    except Exception as e:
        error_msg = f"Error in query_gemini_stream: {str(e)}"
        print(error_msg)
        yield error_msg

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    user_name: Optional[str] = None
    kb_name: Optional[str | List[str]] = None
    is_assamese_english_mixed: Optional[bool] = False

class KnowledgeBaseInfo(BaseModel):
    name: str
    description: str
    is_initialized: bool

# Function to determine if a question is complex enough to warrant WhatsApp group recommendation
def is_complex_question(query: str) -> bool:
    """
    Analyzes a question to determine if it's complex enough to warrant suggesting the WhatsApp group.
    
    Complex questions typically:
    1. Ask about specific admission details not covered in knowledge base
    2. Request personalized advice or guidance
    3. Involve multiple steps or complex processes
    4. Need human intervention to answer accurately
    
    Simple questions that shouldn't trigger WhatsApp suggestion:
    1. Basic facts about the university
    2. General information that should be in our knowledge base
    3. Questions that have straightforward answers
    """
    # Keywords suggesting complexity (admission process, personal guidance, etc.)
    complex_keywords = [
        'admission process', 'how can i apply', 'application deadline', 
        'eligibility criteria', 'entrance exam', 'selection procedure',
        'scholarship', 'financial aid', 'hostel allocation',
        'specialization', 'internship', 'placement', 'career opportunities',
        'industry partners', 'research facilities', 'specific faculty',
        'transfer', 'lateral entry', 'specific course details',
        'syllabus', 'curriculum', 'timetable', 'schedule'
    ]
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check if any complex keywords are in the query
    for keyword in complex_keywords:
        if keyword in query_lower:
            return True
    
    # Check for specific question patterns suggesting complexity
    complex_patterns = [
        r'how (can|do|would) i', r'what (are|is) the process',
        r'when (can|should|will)', r'where (can|should|do)',
        r'which (course|program|branch)', r'requirements?',
        r'specific', r'details? (of|about|for)', r'steps',
        r'btech', r'b\.tech', r'bachelor', r'undergraduate',
        r'specific course', r'admission dates'
    ]
    
    for pattern in complex_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Check query length - longer queries tend to be more complex
    if len(query.split()) > 15:
        return True
    
    # Default to considering it a simple question
    return False

# Add a post-processing function to filter out inappropriate WhatsApp suggestions
def post_process_response(response: str, is_complex: bool) -> str:
    """
    Post-processes the LLM response to filter out inappropriate WhatsApp suggestions
    for simple questions, or to ensure the WhatsApp suggestion isn't the focus of the response.
    """
    # If this is a simple question, remove WhatsApp group suggestions
    if not is_complex:
        # Check if the response contains WhatsApp suggestions
        whatsapp_phrases = [
            "whatsapp group", "join our whatsapp", "whatsapp button", 
            "navbar to join", "whatsapp for more", "join the whatsapp",
            "click the button", "join the group"
        ]
        
        has_whatsapp_suggestion = any(phrase in response.lower() for phrase in whatsapp_phrases)
        
        if has_whatsapp_suggestion:
            # Remove sentences containing WhatsApp suggestions
            sentences = re.split(r'(?<=[.!?])\s+', response)
            filtered_sentences = []
            
            for sentence in sentences:
                if not any(phrase in sentence.lower() for phrase in whatsapp_phrases):
                    filtered_sentences.append(sentence)
            
            # If we removed too much, add a generic closing
            if len(filtered_sentences) < len(sentences) - 2:
                filtered_sentences.append("Is there anything else I can help you with about BSSRV University?")
                
            return " ".join(filtered_sentences)
    
    # For complex questions, ensure WhatsApp suggestion isn't repetitive or the focus
    else:
        whatsapp_count = response.lower().count("whatsapp")
        
        # If WhatsApp is mentioned too many times, simplify
        if whatsapp_count > 1:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            whatsapp_sentences = []
            non_whatsapp_sentences = []
            
            for sentence in sentences:
                if "whatsapp" in sentence.lower():
                    whatsapp_sentences.append(sentence)
                else:
                    non_whatsapp_sentences.append(sentence)
            
            # Keep only the first WhatsApp mention
            if whatsapp_sentences:
                result = non_whatsapp_sentences + [whatsapp_sentences[0]]
                return " ".join(result)
    
    # If no changes needed, return the original response
    return response

# Update the chat endpoint to use Gemini instead of Groq
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Check if query is about agriculture
        if is_agriculture_query(request.query):
            return {"response": "I don't have any information about agriculture programs at BSSRV University. Please ask about our B.Tech programs instead."}
            
        # Special case for "who built you" type questions
        builder_patterns = [
            r"who (made|created|built|developed|designed) you",
            r"who is your (creator|developer|designer|builder|maker)",
            r"who (are you made|were you built|were you created|were you developed) by",
            r"who.s behind you",
            r"who built the (chatbot|bot|assistant)",
            r"who developed the (chatbot|bot|assistant)"
        ]
        
        # Check if the query matches any of the builder patterns
        if any(re.search(pattern, request.query.lower()) for pattern in builder_patterns):
            return {"response": "I was built by Shyamol Konwar, from CSE branch. He developed me as an AI assistant to help students and prospective applicants learn about BSSRV University."}
            
        # Use the new knowledge base name
        kb_name = "bssrv_university_information"
        
        print(f"Chat request received - Query: '{request.query}' | Using knowledge base: {kb_name}")
        
        # Check if the new knowledge base exists
        if kb_name not in knowledge_bases:
            raise HTTPException(status_code=400, detail=f"Knowledge base '{kb_name}' not found")
        
        kb = knowledge_bases[kb_name]
        
        if not kb.vector_store:
            print(f"ERROR: Vector store for {kb_name} is not initialized")
            return {"response": "I apologize, but I don't have that specific information in my knowledge base at the moment. Is there something else I can help you with about BSSRV University?"}
        
        try:
            print(f"Performing similarity search for query in the document")
            
            # Pre-process the query to detect key topics
            query_lower = request.query.lower()
            
            # Check for common topics in the query to optimize retrieval
            is_eligibility_query = any(term in query_lower for term in [
                "eligible", "eligibility", "qualify", "qualification", "criteria", 
                "marks", "percentage", "grade", "cutoff", "get in", "admission", 
                "requirements", "minimum", "class 12", "12th", "higher secondary"
            ])
            
            is_admission_query = any(term in query_lower for term in [
                "admission", "apply", "application", "entrance", "exam", "test", "jee",
                "counseling", "selection", "how to get", "procedure", "process", "enrollment"
            ])
            
            is_program_query = any(term in query_lower for term in [
                "program", "course", "branch", "department", "major", "specialization",
                "btech", "b.tech", "engineering", "cse", "ece", "ai", "ml"
            ])
            
            is_fee_query = any(term in query_lower for term in [
                "fee", "fees", "cost", "tuition", "expense", "payment", "scholarship",
                "financial", "hostel fee", "semester fee", "annual"
            ])
            
            # If specific topic detected, include topic-specific search filter
            mmr_filter = None
            if is_eligibility_query:
                mmr_filter = "eligibility criteria academic qualifications cutoff marks"
            elif is_admission_query:
                mmr_filter = "admission process application procedure selection process"
            elif is_program_query:
                mmr_filter = "program courses branch department specialization btech engineering cse ece"
            elif is_fee_query:
                mmr_filter = "fee structure tuition fee hostel fee semester fee"
            
            # Use MMR for diversity in retrieval when no specific filter is available
            # This helps to retrieve more diverse and relevant chunks
            if mmr_filter:
                # First get topic-specific chunks
                topic_docs = kb.vector_store.similarity_search(
                    request.query + " " + mmr_filter, 
                    k=10
                )
                
                # Then get general chunks related to the query
                general_docs = kb.vector_store.max_marginal_relevance_search(
                    request.query, 
                    k=5,
                    fetch_k=15
                )
                
                # Combine both sets of chunks, removing duplicates
                seen_content = set()
                docs = []
                
                for doc in topic_docs + general_docs:
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        docs.append(doc)
                
                # Limit to top 15 chunks
                docs = docs[:15]
            else:
                # If no specific topic detected, use MMR to ensure diverse results
                docs = kb.vector_store.max_marginal_relevance_search(
                    request.query, 
                    k=15,  # Return 15 chunks
                    fetch_k=25  # Consider 25 chunks for diversity
                )
                
            print(f"Found {len(docs)} relevant chunks in the document")
            
            if not docs:
                print("No relevant information found in the document")
                return {"response": "I apologize, but I don't have that specific information in my knowledge base at the moment. Is there something else I can help you with about BSSRV University?"}
                
            # Create context from retrieved document chunks
            context = "\n".join([doc.page_content for doc in docs])
            print(f"Created context with {len(context)} characters from {len(docs)} document chunks")
            
            # Add detailed logging for diagnostics
            print("Top retrieved chunks (first 100 chars of each):")
            for i, doc in enumerate(docs[:5]):  # Log first 5 chunks
                print(f"Chunk {i+1}: {doc.page_content[:100]}...")
            
            # Create a detailed prompt with clear instructions and emphasis on maintaining conversation context
            prompt = f"""
Context from BSSRV University document:
{context}

User query: {request.query}

Important instructions:
1. Respond to the user's query in a conversational, natural way without phrases like "According to the context" or "Based on the information".
2. Speak as if you're a helpful university assistant having a casual chat.
3. ONLY include information that is EXPLICITLY present in the context - provide specific details directly from the context.
4. When asked about links, facilities, admission details, or anything specific, EXTRACT and provide the EXACT information from the context.
5. MAINTAIN CONVERSATION CONTEXT AT ALL TIMES. If the user's query is a short response like "yes" or "yeah", treat it as continuing the previous conversation topic.
6. If the user responds with a short affirmative like "yes" or "yeah", continue elaborating on the previous topic you were discussing or the question you asked.
7. NEVER END YOUR RESPONSE WITH A QUESTION. If you want to ask a question, do it in the middle of your response, but not at the end.
8. DO NOT ask if the user wants more information, has questions, or needs anything else - wait for them to ask.
9. If the information isn't in the context, simply acknowledge that you don't have that specific information at the moment.
10. ALWAYS remember the entire conversation context when responding. Don't start fresh with each message.
11. ENSURE ACCURACY: Before responding, verify the information in the context. Double-check dates, contact details, and program information.
12. PRIORITIZE FACTUAL INFORMATION: Focus on providing factual information from the context first, before adding conversational elements.
13. BE PRECISE: When asked about specific details like tuition fees, application deadlines, or contact information, provide the exact values from the context.
"""

            # If the query is in Assamese-English mixed language, add instructions to respond similarly
            if request.is_assamese_english_mixed:
                prompt += """
14. LANGUAGE STYLE: The user is using a mix of English and Assamese (Anglo-Assamese). 
    - Respond in a similar mixed language style 
    - DO NOT use Bengali script or any other language script
    - Use common Assamese words like 'moi', 'apuni', 'tumi', 'ase', 'nai', 'loi', 'korim', 'koribo', 'lage' etc.
    - Keep technical terms, college names, and program names in English
15. NEVER respond in Bengali language or script. Assamese mixed with English should be written in English script only.
"""
            
            print("Sending query to Gemini API")
            response = query_gemini(prompt, request.user_name, kb_name, "gemini-2.0-flash", request.is_assamese_english_mixed)
            print(f"Received response from Gemini API: {len(response)} characters")
            
            # Process the response to ensure it maintains context
            processed_response = remove_duplicate_questions(response)
            
            return {"response": processed_response}
        except Exception as e:
            print(f"Error searching document: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Update streaming chat endpoint to use Gemini
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        # Check if query is about agriculture
        if is_agriculture_query(request.query):
            async def agriculture_response_generator():
                response = "BSSRV University does not offer BSc Agriculture or any agriculture-related programs. The university only offers B.Tech programs in various engineering disciplines including Computer Science (CSE), Computer Science with AI/ML, Computer Science and Business Science (CSBS), and Electronics and Communication Engineering (ECE). For information about our B.Tech programs, please ask specifically about those."
                # Convert the message to a character-by-character stream for typewriter effect
                for char in response:
                    yield f"data: {char}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for typewriter effect
            
            return StreamingResponse(
                content=agriculture_response_generator(),
                media_type="text/event-stream"
            )
        
        # Special case for "who built you" type questions
        builder_patterns = [
            r"who (made|created|built|developed|designed) you",
            r"who is your (creator|developer|designer|builder|maker)",
            r"who (are you made|were you built|were you created|were you developed) by",
            r"who.s behind you",
            r"who built the (chatbot|bot|assistant)",
            r"who developed the (chatbot|bot|assistant)"
        ]
        
        # Check if the query matches any of the builder patterns
        if any(re.search(pattern, request.query.lower()) for pattern in builder_patterns):
            async def builder_response_generator():
                response = "I was built by Shyamol Konwar, from CSE branch. He developed me as an AI assistant to help students and prospective applicants learn about BSSRV University."
                # Convert the message to a character-by-character stream for typewriter effect
                for char in response:
                    yield f"data: {char}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for typewriter effect
            
            return StreamingResponse(
                content=builder_response_generator(),
                media_type="text/event-stream"
            )
        
        # Use the new knowledge base name
        kb_name = "bssrv_university_information"
    
        print(f"Chat stream request received - Query: '{request.query}' | Using knowledge base: {kb_name}")
    
        # Check if the new knowledge base exists
        if kb_name not in knowledge_bases:
            print(f"Warning: Knowledge base '{kb_name}' not found")
            
            async def simple_response_generator():
                message = "I apologize, but I don't have that specific information in my knowledge base at the moment. Is there something else I can help you with about BSSRV University?"
                    
                # Convert the message to a character-by-character stream for typewriter effect
                for char in message:
                    yield f"data: {char}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for typewriter effect
            
            return StreamingResponse(
                content=simple_response_generator(),
                media_type="text/event-stream"
            )
            
        kb = knowledge_bases[kb_name]
    
        if not kb.vector_store:
            print(f"ERROR: Vector store for {kb_name} is not initialized")
            
            async def error_response_generator():
                message = "I apologize, but I don't have that specific information in my knowledge base at the moment. Is there something else I can help you with about BSSRV University?"
                    
                # Convert the message to a character-by-character stream for typewriter effect
                for char in message:
                    yield f"data: {char}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for typewriter effect
            
            return StreamingResponse(
                content=error_response_generator(),
                media_type="text/event-stream"
            )
    
        try:
            print(f"Performing similarity search for query in the document")
            
            # Pre-process the query to detect key topics
            query_lower = request.query.lower()
            
            # Check for common topics in the query to optimize retrieval
            is_eligibility_query = any(term in query_lower for term in [
                "eligible", "eligibility", "qualify", "qualification", "criteria", 
                "marks", "percentage", "grade", "cutoff", "get in", "admission", 
                "requirements", "minimum", "class 12", "12th", "higher secondary"
            ])
            
            is_admission_query = any(term in query_lower for term in [
                "admission", "apply", "application", "entrance", "exam", "test", "jee",
                "counseling", "selection", "how to get", "procedure", "process", "enrollment"
            ])
            
            is_program_query = any(term in query_lower for term in [
                "program", "course", "branch", "department", "major", "specialization",
                "btech", "b.tech", "engineering", "cse", "ece", "ai", "ml"
            ])
            
            is_fee_query = any(term in query_lower for term in [
                "fee", "fees", "cost", "tuition", "expense", "payment", "scholarship",
                "financial", "hostel fee", "semester fee", "annual"
            ])
            
            # If specific topic detected, include topic-specific search filter
            mmr_filter = None
            if is_eligibility_query:
                mmr_filter = "eligibility criteria academic qualifications cutoff marks"
            elif is_admission_query:
                mmr_filter = "admission process application procedure selection process"
            elif is_program_query:
                mmr_filter = "program courses branch department specialization btech engineering cse ece"
            elif is_fee_query:
                mmr_filter = "fee structure tuition fee hostel fee semester fee"
            
            # Use MMR for diversity in retrieval when no specific filter is available
            # This helps to retrieve more diverse and relevant chunks
            if mmr_filter:
                # First get topic-specific chunks
                topic_docs = kb.vector_store.similarity_search(
                    request.query + " " + mmr_filter, 
                    k=10
                )
                
                # Then get general chunks related to the query
                general_docs = kb.vector_store.max_marginal_relevance_search(
                    request.query, 
                    k=5,
                    fetch_k=15
                )
                
                # Combine both sets of chunks, removing duplicates
                seen_content = set()
                docs = []
                
                for doc in topic_docs + general_docs:
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        docs.append(doc)
                
                # Limit to top 15 chunks
                docs = docs[:15]
            else:
                # If no specific topic detected, use MMR to ensure diverse results
                docs = kb.vector_store.max_marginal_relevance_search(
                    request.query, 
                    k=15,  # Return 15 chunks
                    fetch_k=25  # Consider 25 chunks for diversity
                )
                
            print(f"Found {len(docs)} relevant chunks in the document")
            
            if not docs:
                print("No relevant information found in the document")
                
                async def no_docs_generator():
                    message = "I apologize, but I don't have that specific information in my knowledge base at the moment. Is there something else I can help you with about BSSRV University?"
                        
                    # Convert the message to a character-by-character stream for typewriter effect
                    for char in message:
                        yield f"data: {char}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for typewriter effect
                
                return StreamingResponse(
                    content=no_docs_generator(),
                    media_type="text/event-stream"
                )
                
            # Create context from retrieved document chunks
            context = "\n".join([doc.page_content for doc in docs])
            print(f"Created context with {len(context)} characters from {len(docs)} document chunks")
            
            # Add detailed logging for diagnostics
            print("Top retrieved chunks (first 100 chars of each):")
            for i, doc in enumerate(docs[:5]):  # Log first 5 chunks
                print(f"Chunk {i+1}: {doc.page_content[:100]}...")
            
            # Create a detailed prompt with clear instructions and emphasis on maintaining conversation context
            prompt = f"""
Context from BSSRV University document:
{context}

User query: {request.query}

Important instructions:
1. Respond to the user's query in a conversational, natural way without phrases like "According to the context" or "Based on the information".
2. Speak as if you're a helpful university assistant having a casual chat.
3. ONLY include information that is EXPLICITLY present in the context - provide specific details directly from the context.
4. When asked about links, facilities, admission details, or anything specific, EXTRACT and provide the EXACT information from the context.
5. MAINTAIN CONVERSATION CONTEXT AT ALL TIMES. If the user's query is a short response like "yes" or "yeah", treat it as continuing the previous conversation topic.
6. If the user responds with a short affirmative like "yes" or "yeah", continue elaborating on the previous topic you were discussing or the question you asked.
7. NEVER END YOUR RESPONSE WITH A QUESTION. If you want to ask a question, do it in the middle of your response, but not at the end.
8. DO NOT ask if the user wants more information, has questions, or needs anything else - wait for them to ask.
9. If the information isn't in the context, simply acknowledge that you don't have that specific information at the moment.
10. ALWAYS remember the entire conversation context when responding. Don't start fresh with each message.
11. If links are requested, PROVIDE THE EXACT LINKS from the context. Application links format is: admission.bssrv.ac.in/eforms/b-tech-2025-26/17/
12. ENSURE ACCURACY: Before responding, verify the information in the context. Double-check dates, contact details, and program information.
13. PRIORITIZE FACTUAL INFORMATION: Focus on providing factual information from the context first, before adding conversational elements.
14. BE PRECISE: When asked about specific details like tuition fees, application deadlines, or contact information, provide the exact values from the context.
"""

            # If the query is in Assamese-English mixed language, add instructions to respond similarly
            if request.is_assamese_english_mixed:
                prompt += """
15. LANGUAGE STYLE: The user is using a mix of English and Assamese (Anglo-Assamese). 
    - Respond in a similar mixed language style 
    - DO NOT use Bengali script or any other language script
    - Use common Assamese words like 'moi', 'apuni', 'tumi', 'ase', 'nai', 'loi', 'korim', 'koribo', 'lage' etc.
    - Keep technical terms, college names, and program names in English
16. NEVER respond in Bengali language or script. Assamese mixed with English should be written in English script only.
"""
            
            print("Starting streaming response from Gemini API")
            
            async def stream_generator():
                try:
                    async for chunk in query_gemini_stream(prompt, request.user_name, kb_name, request.is_assamese_english_mixed):
                        # Format for SSE
                        yield f"data: {chunk}\n\n"
                except Exception as e:
                    error_msg = f"Error in stream generator: {str(e)}"
                    print(error_msg)
                    yield f"data: I'm sorry, I encountered an error while processing your request. Please try again.\n\n"
            
            return StreamingResponse(
                content=stream_generator(),
                media_type="text/event-stream"
            )
        except Exception as e:
            print(f"Error searching document: {str(e)}")
            
            # Return a streaming error response
            async def error_generator():
                yield f"data: I'm sorry, I encountered an error. Please try again later.\n\n"
                
            return StreamingResponse(
                content=error_generator(),
                media_type="text/event-stream"
            )
            
    except Exception as e:
        print(f"Error processing streaming chat request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a streaming error response instead of raising HTTP exception
        async def error_generator():
            yield f"data: I'm sorry, I encountered an error: {str(e)}. Please try again later.\n\n"
            
        return StreamingResponse(
            content=error_generator(),
            media_type="text/event-stream"
        )

# Get available knowledge bases
@app.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def get_knowledge_bases():
    return [
        KnowledgeBaseInfo(
            name=kb.name,
            description=kb.description,
            is_initialized=kb.vector_store is not None
        )
        for kb in knowledge_bases.values()
    ]

# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "knowledge_bases": {
            name: kb.vector_store is not None
            for name, kb in knowledge_bases.items()
        }
    }

class GeminiRequest(BaseModel):
    query: str
    model: Optional[str] = "gemini-2.0-flash"

# Add a new endpoint to test Gemini
@app.post("/generate/gemini")
async def generate_with_gemini(request: GeminiRequest):
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        print(f"Generating content with Gemini model: {request.model}")
        
        model = genai.GenerativeModel(request.model)
        response = model.generate_content(request.query)
        return {"response": response.text}
    except Exception as e:
        print(f"Error generating content with Gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")