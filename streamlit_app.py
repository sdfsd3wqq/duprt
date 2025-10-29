import streamlit as st
import google.generativeai as genai
from google.generativeai import types
from google.api_core.exceptions import APIError
import os
import PyPDF2
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from pydantic import BaseModel, Field

# --- Pydantic Schemas for Structured Output ---

# 1. Structured Schema for Prioritization Matrix
class Recommendation(BaseModel):
    """A single strategic recommendation with impact and effort scores."""
    recommendation: str = Field(description="The strategic initiative or recommendation.")
    impact: str = Field(description="Potential business impact (High, Medium, or Low).")
    effort: str = Field(description="Implementation effort/cost (High, Medium, or Low).")

class PrioritizationOutput(BaseModel):
    """The complete structured output for prioritization."""
    recommendations_data: list[Recommendation] = Field(description="A list of all analyzed recommendations with their scores.")
    textual_analysis: str = Field(description="A comprehensive textual analysis of the results, categorizing them into Quick Wins, Major Projects, Fill-ins, and Avoid. Provide clear reasoning.")

# 2. Structured Schema for Hypothesis Generation (Simple list)
class HypothesisList(BaseModel):
    """A list of testable hypotheses."""
    hypotheses: list[str] = Field(description="A list of 5-7 testable business hypotheses related to the problem.")


# --- [PORTED FROM FLASK APP] Code Validation & Parsing Logic ---

class CodeValidator:
    """Enhanced class to validate and fix generated code"""

    @staticmethod
    def validate_and_fix_html(html_code: str) -> str:
        """Ensure HTML has proper structure and is complete"""
        if not html_code.strip():
            return CodeValidator.get_minimal_html()
        
        # Check for basic HTML structure
        if '<!DOCTYPE html>' not in html_code:
            html_code = '<!DOCTYPE html>\n' + html_code
        
        # Ensure proper HTML structure
        if '<html' not in html_code:
            html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Website</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    {html_code}
    <script src="script.js"></script>
</body>
</html>"""
        
        # Ensure closing tags are present
        html_code = CodeValidator.fix_html_structure(html_code)
        
        # Ensure CSS and JS links are present
        if 'styles.css' not in html_code:
            if '</head>' in html_code:
                html_code = html_code.replace('</head>', '    <link rel="stylesheet" href="styles.css">\n</head>')
            elif '<head>' in html_code:
                html_code = html_code.replace('<head>', '<head>\n    <link rel="stylesheet" href="styles.css">')
        
        if 'script.js' not in html_code:
            if '</body>' in html_code:
                html_code = html_code.replace('</body>', '    <script src="script.js"></script>\n</body>')
            elif '<body>' in html_code:
                html_code += '\n    <script src="script.js"></script>\n</body>'
        
        return html_code

    @staticmethod
    def fix_html_structure(html_code: str) -> str:
        """Fix common HTML structure issues"""
        # Ensure body tag exists
        if '<body>' not in html_code:
            # Find where to insert body tag
            if '</head>' in html_code:
                html_code = html_code.replace('</head>', '</head>\n<body>') + '</body>'
            elif '<html>' in html_code:
                html_code = html_code.replace('<html>', '<html>\n<body>') + '</body>'
        
        # Ensure head tag exists
        if '<head>' not in html_code:
            if '<html>' in html_code:
                html_code = html_code.replace('<html>', '<html>\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Generated Website</title>\n</head>')
        
        # Fix unclosed tags
        tags_to_close = ['<header', '<main', '<section', '<div', '<nav', '<form']
        for tag in tags_to_close:
            if f'<{tag}' in html_code and f'</{tag.split()[0]}' not in html_code:
                # Add closing tag at the end of body
                if '</body>' in html_code:
                    html_code = html_code.replace('</body>', f'</{tag.split()[0]}>\n</body>')
        
        return html_code

    @staticmethod
    def validate_and_fix_css(css_code: str) -> str:
        """Ensure CSS is valid and complete"""
        if not css_code.strip():
            return CodeValidator.get_minimal_css()
        
        # Ensure basic responsive design
        if '@media' not in css_code:
            css_code += """
/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    .grid {
        grid-template-columns: 1fr;
    }
    
    nav ul {
        flex-direction: column;
    }
}"""
        
        # Ensure basic reset
        if 'box-sizing' not in css_code:
            css_code = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

""" + css_code
        
        return css_code

    @staticmethod
    def validate_and_fix_js(js_code: str) -> str:
        """Ensure JavaScript is valid and has error handling"""
        if not js_code.strip():
            return CodeValidator.get_minimal_js()
        
        # Add basic error handling if missing
        if 'try' not in js_code and 'catch' not in js_code:
            js_code = """
// Error handling wrapper
function safeExecute(callback, errorMessage = 'An error occurred') {
    try {
        return callback();
    } catch (error) {
        console.error(errorMessage, error);
        return null;
    }
}

""" + js_code
        
        # Ensure DOM ready event
        if 'DOMContentLoaded' not in js_code and 'document.addEventListener' not in js_code:
            js_code = f"""
document.addEventListener('DOMContentLoaded', function() {{
    // Initialize website functionality
    {js_code}
}});"""
        
        return js_code

    @staticmethod
    def get_minimal_html() -> str:
        """
        NEW: Return an error message instead of a template.
        This makes it clear the AI generation failed.
        """
        return """
<body>
    <header><h1>Generation Error</h1></header>
    <main>
        <section>
            <h2>Presentation Generation Failed</h2>
            <p>The AI model did not return valid HTML code. Please try again.</p>
        </section>
    </main>
</body>
</html>"""

    @staticmethod
    def get_minimal_css() -> str:
        """
        NEW: Return minimal error styling instead of a template.
        """
        return """
body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
header { background: #e74c3c; color: white; padding: 1rem; text-align: center; }
main { max-width: 800px; margin: 1rem auto; padding: 1rem; border: 1px solid #ddd; }
h2 { color: #e74c3c; }
"""

    @staticmethod
    def get_minimal_js() -> str:
        """
        NEW: Return a simple console log for the error state.
        """
        return """
document.addEventListener('DOMContentLoaded', function() {
    console.error('Presentation failed to load: AI did not generate valid JavaScript.');
});"""

def parse_code(full_response: str) -> dict[str, str]:
    """
    Parse the Gemini response for code blocks with enhanced error handling and validation.
    """
    patterns = {
        'html': r'```html\n(.*?)\n```',
        'css': r'```css\n(.*?)\n```',
        'js': r'```(?:javascript|js)\n(.*?)\n```'
    }
    
    codes = {}
    validator = CodeValidator()
    
    for lang, pattern in patterns.items():
        try:
            match = re.search(pattern, full_response, re.DOTALL)
            if match:
                code_content = match.group(1).strip()
                
                # Validate and fix code structure
                if lang == 'html':
                    code_content = validator.validate_and_fix_html(code_content)
                elif lang == 'css':
                    code_content = validator.validate_and_fix_css(code_content)
                elif lang == 'js':
                    code_content = validator.validate_and_fix_js(code_content)
                
                codes[lang] = code_content
            else:
                # Use fallback (now an error message) if code block is missing
                if lang == 'html':
                    codes[lang] = validator.get_minimal_html()
                elif lang == 'css':
                    codes[lang] = validator.get_minimal_css()
                else:
                    codes[lang] = validator.get_minimal_js()
                    
        except Exception as e:
            st.error(f"Error parsing {lang} code: {str(e)}")
            if lang == 'html':
                codes[lang] = validator.get_minimal_html()
            elif lang == 'css':
                codes[lang] = validator.get_minimal_css()
            else:
                codes[lang] = validator.get_minimal_js()

    return codes

# --- END of Ported Logic ---


# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Use st.error and stop as before
    st.error("GEMINI_API_KEY not found. Please create a .env file and add it.")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# --- Advanced Configuration ---
st.set_page_config(
    page_title="Strategic Consulting AI", 
    layout="wide",
    page_icon="🧠"
)

# --- Session State Initialization ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None
if 'global_context' not in st.session_state: # NEW: Centralized Context
    st.session_state.global_context = ""
if 'generated_presentation' not in st.session_state: # NEW: For slide display
    st.session_state.generated_presentation = None
if 'current_analysis_summary' not in st.session_state: # NEW: To hold analysis for presentation
    st.session_state.current_analysis_summary = ""
if 'presentation_prompt' not in st.session_state: # NEW: Store custom presentation prompt
    st.session_state.presentation_prompt = ""


# --- Consolidated Gemini Helper Function with Structured Output ---
def get_gemini_response(prompt_text, model_name='gemini-2.5-flash', response_schema=None, files=None):
    """A robust helper function to call the Gemini API with optional files and structured output."""
    try:
        config = types.GenerateContentConfig()
        
        # Configure Structured Output
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema
            
        # Prepare contents (prompt_text + files/data)
        contents = [prompt_text]
        if files:
            contents.extend(files)

        # Determine the correct model name
        # Use 1.5-flash by default, but allow override for powerful models
        effective_model_name = model_name

        response = client.models.generate_content(
            model=effective_model_name,
            contents=contents,
            config=config
        )
        
        # Parse structured output if schema was used
        if response_schema:
            return json.loads(response.text)
        
        return response.text
    except APIError as e:
        st.error(f"Gemini API Error ({model_name}): {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Structured Output JSON Error: {e}\nRaw Response: {response.text[:500]}...")
        return None
    except Exception as e:
        st.error(f"General Gemini Error: {e}")
        return None


# --- [IMPROVED] Presentation Generation Function with Custom Prompt Support ---
def generate_presentation_slides(analysis_content: str, custom_prompt: str = None):
    """
    Uses the website generator logic to create HTML/CSS/JS slides.
    This function now supports custom user prompts for presentation generation.
    """
    
    # Base prompt for presentation generation
    base_prompt = """
    You are an elite-level AI creative agent, a specialist in both strategic communication and full-stack web development (UI/UX, HTML, CSS, JS).
    Your task is to generate a complete, interactive, and visually stunning single-page HTML/CSS/JS slide deck *from scratch*.
    
    **CRITICAL REQUIREMENT: DO NOT USE ANY TEMPLATES OR PRE-EXISTING DESIGNS.** You must invent a unique, professional design aesthetic (color palette, typography, layout) that is *themed* to match the analysis content provided.
    
    **ANALYSIS CONTENT TO VISUALIZE:**
    "{analysis_content}"
    
    **DESIGN & STRUCTURE REQUIREMENTS:**

    1.  **Parse and Structure:**
        * First, deeply analyze the `ANALYSIS CONTENT`.
        * Then, structure this content into a logical slide-based presentation. This must include:
            * A compelling **Title Slide** (e.g., "Strategic Analysis: [Project Name]").
            * An **Executive Summary** slide.
            * Multiple **Content Slides** (e.g., "Key Findings," "Problem Breakdown," "Hypothesis 1 Results," "Prioritization Matrix," "Key Recommendations"). Use lists, blockquotes, and other elements.
            * A concluding **Next Steps** or **Summary** slide.

    2.  **Generate COMPLETE, SEPARATE Code Blocks:** You must provide three distinct, complete, and valid code blocks: one for HTML, one for CSS, and one for JavaScript.
    
    3.  **HTML (index.html):**
        * Use semantic HTML.
        * Create a main container (`<main id="presentation-container">`).
        * Each slide must be a `<section class="slide">` element.
        * The *first* slide should have the class `slide active` to be visible on load.
        * Create navigation buttons: `<button id="prev-slide">Previous</button>` and `<button id="next-slide">Next</button>`.
        * Create a slide counter: `<div id="slide-counter">1 / 5</div>`.

    4.  **CSS (styles.css):**
        * **No simple black text on white.** Create a professional, modern, and *themed* design. (e.g., if the analysis is financial, use blues/greens; if creative, use more vibrant colors).
        * Style the slides to be full-viewport or a large, centered element.
        * **Crucially:** Only the `.slide.active` class should be visible (`display: block` or `opacity: 1`). All other `.slide` elements must be hidden (`display: none` or `opacity: 0`).
        * Add smooth transitions (e.g., `transition: opacity 0.5s ease;`) for slide changes.
        * Style the navigation buttons and slide counter.
        * Ensure it's responsive (`@media` queries).

    5.  **JavaScript (script.js):**
        * Wrap all code in `document.addEventListener('DOMContentLoaded', ...)`
        * Implement the core slide navigation logic:
            * Get all slides (`document.querySelectorAll('.slide')`).
            * Get buttons and counter (`document.getElementById(...)`).
            * Write a function `showSlide(index)` that removes `active` from the current slide, adds it to the new slide, and updates the slide counter.
            * Add click event listeners for 'next-slide' and 'prev-slide' buttons.
        * **Advanced Feature:** Add keyboard navigation (e.g., `document.addEventListener('keydown', ...)` for `ArrowLeft` and `ArrowRight`).
        * Ensure the logic is robust (e.g., doesn't go past the last slide or before the first).

    **CREATIVITY REQUIREMENT:**
    - Invent a unique visual theme that matches the content's tone and industry
    - Use creative layouts beyond simple bullet points (grids, cards, visual hierarchies)
    - Implement subtle animations or interactive elements where appropriate
    - Ensure the design feels premium and professional

    **OUTPUT FORMAT (MANDATORY):**
    ```html
    [COMPLETE, VALID HTML CODE]
    ```
    ```css
    [COMPLETE, VALID CSS CODE]
    ```
    ```javascript
    [COMPLETE, VALID JAVASCRIPT CODE]
    ```
    """
    
    # Use custom prompt if provided, otherwise use base prompt
    if custom_prompt:
        final_prompt = f"""
        {custom_prompt}
        
        **ANALYSIS CONTENT TO VISUALIZE:**
        {analysis_content}
        
        **OUTPUT FORMAT REQUIREMENT:**
        You MUST provide three separate code blocks in this exact format:
        ```html
        [Your complete HTML code here]
        ```
        ```css
        [Your complete CSS code here]
        ```
        ```javascript
        [Your complete JavaScript code here]
        ```
        """
    else:
        final_prompt = base_prompt.format(analysis_content=analysis_content)
    
    with st.spinner("Generating bespoke presentation code with Gemini 1.5 Pro..."):
        # Use the powerful model for this creative task
        full_response = get_gemini_response(final_prompt, 'gemini-2.5-pro') 
        
        if not full_response:
            st.error("Failed to get a response from the AI model.")
            return None
            
    with st.spinner("Validating and parsing presentation code..."):
        codes = parse_code(full_response)
        
        if "Generation Error" in codes.get('html', ''):
            st.error("AI failed to generate valid code. Please try again.")
            st.code(full_response, language='text') # Show the raw response for debugging
            return None
        
        if not codes.get('html') or not codes.get('css') or not codes.get('js'):
            st.error("Generated code was incomplete (missing HTML, CSS, or JS). Please try again.")
            return None
            
    return codes


# --- Chat Functionality (remains similar, uses 'client' from setup) ---
def initialize_chat():
    """Initialize a new chat session"""
    try:
        # Use gemini-1.5-flash-latest for enhanced conversational capability
        st.session_state.current_chat = client.chats.create(model="gemini-2.5-flash") 
    except Exception as e:
        st.error(f"Failed to initialize chat: {e}")

def send_chat_message(message):
    """Send message to chat and get response"""
    if st.session_state.current_chat is None:
        initialize_chat()
    
    try:
        response = st.session_state.current_chat.send_message(message)
        return response.text
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None


# --- Consulting Framework Functions with Structured Output ---

def build_issue_tree(problem_statement):
    """Build a MECE issue tree for problem structuring"""
    prompt = f"""
    You are an expert problem-solver. Structure this problem using systematic decomposition principles.
    
    Problem: {problem_statement}
    
    Create a comprehensive issue tree with 2-3 main branches that break down the problem into smaller, solvable components.
    Format your response with clear headings and bullet points.
    
    For each sub-branch, suggest what data or analysis would be needed to address it.
    """
    return get_gemini_response(prompt, 'gemini-2.5-pro') # Upgraded for better structure

def generate_hypotheses(problem_statement):
    """Generate a list of testable business hypotheses using structured output."""
    prompt = f"""
    You are a strategic consultant. Based on this core problem, generate 5-7 distinct, testable, and mutually exclusive hypotheses that could explain the problem or suggest a solution path.
    
    CORE PROBLEM: {problem_statement}
    
    Format the output strictly according to the provided JSON schema.
    """
    return get_gemini_response(prompt, 'gemini-2.5-flash-latest', response_schema=HypothesisList)

def test_hypothesis(hypothesis, data_context):
    """Test a business hypothesis against available data context."""
    prompt = f"""
    You are a hypothesis-driven analyst. Test this business hypothesis:
    
    HYPOTHESIS: {hypothesis}
    
    AVAILABLE DATA CONTEXT: {data_context}
    
    Analyze whether the hypothesis is supported, refuted, or requires more data.
    
    Structure your response with the following explicit, bolded categories at the start of your response, followed by a detailed markdown analysis:

    **Confidence Level:** [High|Medium|Low]
    **Support/Refute:** [Supported|Refuted|Inconclusive]
    **Recommended Next Step:** [Specific, actionable next step based on the findings]

    Provide a detailed markdown analysis that includes:
    1. Key tests needed to validate the hypothesis
    2. What the available data suggests
    3. Alternative explanations to consider
    
    Be rigorous and evidence-based in your assessment.
    """
    return get_gemini_response(prompt, 'gemini-2.5-pro') # Upgraded for better reasoning

def prioritize_recommendations(recommendations):
    """Create a prioritization matrix for strategic recommendations using Pydantic schema."""
    prompt = f"""
    You are a strategic prioritization expert. Analyze these recommendations, which are separated by newlines or commas. For each recommendation, assess its **Potential Impact** (High/Medium/Low) and **Implementation Effort** (High/Medium/Low).
    
    Recommendations:
    {recommendations}
    
    Provide your analysis and the structured data in a single JSON object that conforms strictly to the provided Pydantic schema.
    """
    # Use the Pydantic schema here
    return get_gemini_response(prompt, 'gemini-2.5-flash', response_schema=PrioritizationOutput)


# --- Plotting Function (Modified for Pydantic output) ---
def create_prioritization_plot(recommendations_data):
    """Create a 2x2 prioritization matrix visualization from structured data."""
    if not recommendations_data:
        return None
    
    categories = {
        'Quick Wins': {'x': [], 'y': [], 'text': [], 'hover': []},
        'Major Projects': {'x': [], 'y': [], 'text': [], 'hover': []},
        'Fill-ins': {'x': [], 'y': [], 'text': [], 'hover': []},
        'Avoid': {'x': [], 'y': [], 'text': [], 'hover': []}
    }
    
    effort_map = {'Low': 1, 'Medium': 2, 'High': 3}
    impact_map = {'Low': 1, 'Medium': 2, 'High': 3}
    point_index = 1 

    # Ensure recommendations_data is a list of dictionaries/Pydantic models
    for rec in recommendations_data:
        # Pydantic models (if used) can be converted to dict with .model_dump() or treated as dicts if loaded from JSON
        rec_dict = rec if isinstance(rec, dict) else rec.model_dump()
        
        effort_str = rec_dict.get('effort', 'Medium').strip().capitalize()
        impact_str = rec_dict.get('impact', 'Medium').strip().capitalize()

        effort = effort_map.get(effort_str, 2)
        impact = impact_map.get(impact_str, 2)
        text = rec_dict.get('recommendation', 'Unnamed Rec')
        
        # Quadrant logic (Midpoint is 2)
        if impact >= 2.0 and effort < 2.0:
            category = 'Quick Wins'
        elif impact >= 2.0 and effort >= 2.0:
            category = 'Major Projects'
        elif impact < 2.0 and effort < 2.0:
            category = 'Fill-ins'
        else:
            category = 'Avoid'
            
        categories[category]['x'].append(effort)
        categories[category]['y'].append(impact)
        categories[category]['text'].append(str(point_index))
        categories[category]['hover'].append(f"#{point_index}: {text}<br>Impact: {impact_str}<br>Effort: {effort_str}")
        point_index += 1
    
    # Plotting setup remains the same
    fig = go.Figure()
    colors = {'Quick Wins': 'green', 'Major Projects': 'blue', 'Fill-ins': 'orange', 'Avoid': 'red'}
    
    for category, data in categories.items():
        if data['x']:
            fig.add_trace(go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers+text',
                name=category,
                marker=dict(size=15, color=colors[category]),
                text=data['text'],
                textposition="middle center",
                hovertext=data['hover'],
                hoverinfo='text'
            ))
    
    # Add quadrant lines and labels (center at 2)
    fig.add_shape(type="line", x0=2, y0=0.5, x1=2, y1=3.5, line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0.5, y0=2, x1=3.5, y1=2, line=dict(color="black", width=1, dash="dash"))
    
    fig.update_layout(
        title="Strategic Prioritization Matrix (Impact vs. Effort)",
        xaxis_title="Implementation Effort",
        yaxis_title="Potential Impact",
        xaxis=dict(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        yaxis=dict(range=[0.5, 3.5], tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High']),
        showlegend=True,
        width=650,
        height=550
    )
    
    # Add quadrant labels
    fig.add_annotation(x=1.25, y=3.25, text="Quick Wins (Go!)", showarrow=False, font=dict(color="darkgreen", size=12))
    fig.add_annotation(x=2.75, y=3.25, text="Major Projects (Plan)", showarrow=False, font=dict(color="darkblue", size=12))
    fig.add_annotation(x=1.25, y=0.75, text="Fill-ins (Delegate)", showarrow=False, font=dict(color="darkorange", size=12))
    fig.add_annotation(x=2.75, y=0.75, text="Avoid (Drop)", showarrow=False, font=dict(color="darkred", size=12))
    
    return fig


# --- Analysis Functions (remain mostly the same) ---
def get_advanced_analysis(context, query, perspective="realist"):
    """Advanced analysis with multiple thinking frameworks."""
    perspective_prompts = {
        "optimist": "Provide an optimistic analysis focusing on opportunities and positive outcomes.",
        "pessimist": "Provide a cautious analysis focusing on risks and potential failures.",
        "realist": "Provide a balanced, evidence-based analysis of likely outcomes.",
        "idealist": "Provide an analysis based on ideal scenarios and theoretical best cases.",
        "pragmatist": "Provide a practical analysis focused on actionable steps and immediate concerns.",
        "cynic": "Provide a skeptical analysis questioning assumptions and highlighting potential flaws."
    }
    
    prompt = f"""
    You are a strategic consulting AI trained in multiple analytical frameworks.
    
    CONTEXT:
    {context}
    
    USER QUERY:
    {query}
    
    ANALYTICAL PERSPECTIVE: {perspective_prompts[perspective]}
    
    Please provide a comprehensive analysis that includes:
    1. Core Problem/Framework Analysis
    2. Key Assumptions & Their Validity
    3. Multiple Scenarios (Best Case/Worst Case/Most Likely)
    4. Strategic Recommendations
    5. Risk Assessment & Mitigation
    6. Key Metrics to Track
    
    Structure your response clearly with headings and bullet points.
    """
    return get_gemini_response(prompt, 'gemini-2.5-pro') # Upgraded for better analysis


def generate_multiple_perspectives(context, query):
    """Generate analysis from a subset of thinking perspectives."""
    perspectives = ["realist", "optimist", "pessimist"] 
    results = {}
    
    for perspective in perspectives:
        with st.spinner(f"Generating {perspective} analysis..."):
            analysis = get_advanced_analysis(context, query, perspective)
            if analysis:
                results[perspective] = analysis
    return results


# --- File Processing Functions (Modified for GenAI Parts) ---

def extract_pdf_text_and_part(uploaded_file):
    """Extracts text for context and creates a genai.types.Part object for the model."""
    text = ""
    try:
        # Reset file pointer to the beginning for PyPDF2
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        
        # Create Part for Gemini API (re-read file content)
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        pdf_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf"
        )
        return text, pdf_part
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None, None


def process_csv_data(uploaded_file):
    """Process CSV data and return DataFrame, summary, and a text Part."""
    try:
        # Reset file pointer to the beginning for pandas
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # Create a text summary for the AI to ingest
        data_summary = f"Dataset columns: {list(df.columns)}\nData Head:\n{df.head().to_string()}"
        
        # Create a Part from the summary (or upload the file itself for Data API, but we stick to text for this demo)
        csv_part = types.Part.from_text(data_summary)

        return df, csv_part, data_summary
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None, None


# --- [IMPROVED] Helper to display presentation with custom prompt options ---
def display_presentation_output():
    """Shows the presentation generation button and the presentation itself with custom prompt options."""
    st.markdown("---")
    st.subheader("📊 AI Presentation Generator")
    
    # Presentation customization options
    with st.expander("🎨 Customize Presentation Generation", expanded=False):
        st.markdown("**Customize how the AI generates your presentation:**")
        
        # Option 1: Use default prompt
        use_custom = st.checkbox("Use custom presentation prompt", value=False)
        
        if use_custom:
            st.session_state.presentation_prompt = st.text_area(
                "Custom Presentation Prompt:",
                value=st.session_state.presentation_prompt or """Create a professional slide deck with these specifications:

- Design Style: Modern, clean, business professional
- Color Scheme: Blues and grays with accent colors
- Layout: Clean, minimal, with ample white space
- Special Requirements: Include data visualization placeholders, use corporate-friendly fonts
- Interactive Elements: Smooth transitions between slides
- Content Organization: Executive summary, key findings, recommendations, next steps

Generate complete HTML, CSS, and JavaScript code for an interactive slide deck.""",
                height=200,
                help="Customize exactly how you want the presentation to be generated. Be specific about design, layout, and features."
            )
            st.info("💡 **Tip:** Be specific about design style, color schemes, layout preferences, and any special features you want.")
        else:
            st.session_state.presentation_prompt = ""
            st.info("Using default AI presentation generation with creative, template-free design.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Generate AI Presentation Slides", key="generate_slides", use_container_width=True):
            if st.session_state.current_analysis_summary:
                with st.spinner("Creating your bespoke presentation..."):
                    slide_code = generate_presentation_slides(
                        st.session_state.current_analysis_summary, 
                        st.session_state.presentation_prompt if st.session_state.presentation_prompt else None
                    )
                    if slide_code:
                        # Combine and display
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <style>{slide_code['css']}</style>
                        </head>
                        <body>
                            {slide_code['html']}
                            <script>{slide_code['js']}</script>
                        </body>
                        </html>
                        """
                        st.session_state.generated_presentation = full_html
                        st.success("🎉 Bespoke AI presentation generated successfully!")
                    else:
                        st.error("❌ Failed to generate presentation slides. Please try again.")
            else:
                st.warning("⚠️ No analysis was performed. Please run an analysis tool first.")

    with col2:
        if st.session_state.generated_presentation:
            if st.button("🗑️ Clear Presentation", key="clear_slides", use_container_width=True):
                st.session_state.generated_presentation = None
                st.rerun()

    # Display the generated presentation
    if st.session_state.generated_presentation:
        st.markdown("---")
        st.subheader("🎬 Your AI-Generated Presentation")
        st.info("💡 **Navigation:** Use the Previous/Next buttons or keyboard arrow keys to navigate through slides.")
        
        st.components.v1.html(st.session_state.generated_presentation, height=600, scrolling=True)
        
        # Download option for the presentation
        st.download_button(
            label="💾 Download Presentation Code",
            data=st.session_state.generated_presentation,
            file_name="ai_presentation.html",
            mime="text/html",
            use_container_width=True
        )


# --- Streamlit App Interface ---
st.title("🧠 Advanced Strategic Consulting AI")
st.markdown("""
**Professional consulting frameworks powered by advanced AI analysis (Gemini 1.5 Pro/Flash with Structured Output)**
""")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")

# NEW: Tool Grouping for better UX
tool_group = st.sidebar.radio(
    "Choose Project Phase:",
    ["1. Project Structuring", "2. Deep Analysis & Data", "3. Prioritization & Delivery", "4. Live Consultation"],
    key="tool_group_choice",
    on_change=lambda: st.session_state.update(generated_presentation=None, current_analysis_summary="") # Clear presentation on tool change
)

tool_map = {
    "1. Project Structuring": {
        "Problem Structuring Workbench": "Problem Structuring Workbench",
        "Hypothesis Testing & Generation": "Hypothesis Testing"
    },
    "2. Deep Analysis & Data": {
        "Multi-Perspective Analysis": "Multi-Perspective Analysis",
        "Strategic Frameworks": "Strategic Frameworks",
        "Document Intelligence (RAG)": "Document Intelligence",
        "Data Analysis Suite": "Data Analysis Suite"
    },
    "3. Prioritization & Delivery": {
        "Strategic Prioritization": "Strategic Prioritization",
        "Conversation History": "Conversation History"
    },
    "4. Live Consultation": {
        "Live Chat Consultation": "Live Chat Consultation"
    }
}

tool_choice_keys = list(tool_map[tool_group].keys())
tool_choice_display = st.sidebar.selectbox(
    "Select Tool:", 
    tool_choice_keys,
    key="tool_choice_display"
)
tool_choice = tool_map[tool_group][tool_choice_display]


# --- Advanced File Upload Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Data Integration")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents/Data", 
    type=['pdf', 'csv', 'txt'], 
    accept_multiple_files=True
)

# Process uploaded files (Modified to save Part objects)
if uploaded_files:
    # Use a flag to avoid unnecessary reruns if files haven't changed
    new_files_uploaded = False
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_data:
            new_files_uploaded = True
            
            if file.type == "application/pdf":
                text, part = extract_pdf_text_and_part(file)
                if text and part:
                    st.session_state.uploaded_data[file.name] = {
                        'type': 'pdf', 'content': text, 'part': part, 'size': len(text)
                    }
                    st.sidebar.success(f"✅ {file.name} loaded (PDF)")
            
            elif file.type == "text/csv":
                df, part, summary = process_csv_data(file)
                if df is not None:
                    st.session_state.uploaded_data[file.name] = {
                        'type': 'csv', 'content': df, 'part': part, 'summary': summary, 'size': df.shape
                    }
                    st.sidebar.success(f"✅ {file.name} loaded (CSV)")
            
            elif file.type == "text/plain":
                file.seek(0)
                text = file.getvalue().decode("utf-8")
                part = types.Part.from_text(text)
                st.session_state.uploaded_data[file.name] = {
                    'type': 'txt', 'content': text, 'part': part, 'size': len(text)
                }
                st.sidebar.success(f"✅ {file.name} loaded (TXT)")
    
    # Rerun only if new files were processed successfully
    if new_files_uploaded:
        st.rerun()

# NEW: Global Context Editor
st.sidebar.markdown("---")
st.sidebar.subheader("Global Project Context")
st.session_state.global_context = st.sidebar.text_area(
    "Set high-level context here:",
    value=st.session_state.global_context,
    height=100,
    placeholder="e.g., The company is a B2B SaaS provider facing 15% churn and aggressive competition in the FinTech space."
)
if st.session_state.global_context:
    st.sidebar.success("Global Context Set.")


# --- TOOL 1: Problem Structuring Workbench ---
if tool_choice == "Problem Structuring Workbench":
    st.header("🔄 Problem Structuring Workbench")
    st.markdown("**Structure complex problems using a MECE (Mutually Exclusive, Collectively Exhaustive) Issue Tree.**")
    
    problem_statement = st.text_area(
        "Enter your core problem statement (or use Global Context):",
        height=100,
        value=st.session_state.global_context,
        placeholder="e.g., How can we improve customer satisfaction scores?"
    )
    
    if st.button("Build Problem Structure", type="primary"):
        if problem_statement:
            with st.spinner("Structuring your problem using systematic decomposition..."):
                issue_tree = build_issue_tree(problem_statement) 	
                if issue_tree:
                    st.subheader("📊 Problem Structure (MECE Issue Tree)")
                    st.markdown(issue_tree)
                    
                    # NEW: Save for presentation
                    st.session_state.current_analysis_summary = f"Problem Structure for: {problem_statement}\n\n{issue_tree}"
                    st.session_state.generated_presentation = None # Clear old presentation
                    
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Problem Structuring',
                        'problem': problem_statement,
                        'structure': issue_tree
                    })
                    st.success("Problem structure created successfully!")
        else:
            st.warning("Please enter a problem statement to structure.")

    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 2: Hypothesis Testing (Enhanced with Generation) ---
elif tool_choice == "Hypothesis Testing":
    st.header("🔍 Hypothesis Testing & Validation")
    st.markdown("**Test business hypotheses with data-driven validation and evidence-based assessment.**")
    
    # NEW: Hypothesis Generation
    st.subheader("💡 1. Generate Hypotheses")
    problem_statement = st.text_area("Problem to Generate Hypotheses for:", 
                                         value=st.session_state.global_context, 
                                         height=80)
    
    if st.button("Generate Testable Hypotheses"):
        if problem_statement:
            with st.spinner("Generating core hypotheses..."):
                hypothesis_list_dict = generate_hypotheses(problem_statement)
                if hypothesis_list_dict and 'hypotheses' in hypothesis_list_dict:
                    st.session_state.hypotheses_generated = hypothesis_list_dict['hypotheses']
                    st.subheader("Generated Hypotheses:")
                    for i, h in enumerate(st.session_state.hypotheses_generated):
                        st.markdown(f"**{i+1}.** {h}")
                else:
                    st.error("Could not generate hypotheses. Try again.")
    
    st.markdown("---")
    st.subheader("🧪 2. Test a Hypothesis")
    
    # Use generated hypotheses if available
    hypothesis_options = ["(Enter Custom Hypothesis)"]
    if 'hypotheses_generated' in st.session_state:
        hypothesis_options.extend(st.session_state.hypotheses_generated)
        
    selected_hypothesis_text = st.selectbox("Select or Enter Hypothesis:", hypothesis_options)
    
    if selected_hypothesis_text == "(Enter Custom Hypothesis)":
        hypothesis = st.text_area(
            "Enter your custom hypothesis:",
            height=80,
            placeholder="e.g., Sales decline is due to increased competition."
        )
    else:
        hypothesis = selected_hypothesis_text
    
    st.subheader("Data/Context for Testing")
    data_context = ""
    data_files = []
    
    # Consolidate selected data context (using Part objects now)
    if st.session_state.uploaded_data:
        st.markdown("Select uploaded files to use for context:")
        for filename, file_data in st.session_state.uploaded_data.items():
            if st.checkbox(f"Use {filename} ({file_data['type'].upper()})", key=f"hyp_use_{filename}"):
                data_files.append(file_data['part'])
                if file_data['type'] == 'csv':
                    data_context += f"\n- {filename} (CSV, summary of columns available)."
                else:
                    data_context += f"\n- {filename} (Document, content available for analysis)."
    
    manual_context = st.text_input("Or add manual context:", placeholder="e.g., Sales data Q1-Q3 shows a 10% drop.")
    data_context += "\n" + manual_context

    if st.button("Test Hypothesis", type="primary"):
        if hypothesis and data_context:
            with st.spinner("Testing hypothesis with rigorous analysis..."):
                result = test_hypothesis(hypothesis, data_context)
                
                if result:
                    # --- Extract structured summary metrics (Regex remains for this output) ---
                    confidence = re.search(r'\*\*Confidence Level:\*\* \s*([^\n]+)', result)
                    support = re.search(r'\*\*Support/Refute:\*\* \s*([^\n]+)', result)
                    next_step = re.search(r'\*\*Recommended Next Step:\*\* \s*([^\n]+)', result)

                    st.subheader("🧪 Hypothesis Test Results Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Confidence Level", confidence.group(1).strip() if confidence else "N/A")
                    with col2: st.metric("Support/Refute", support.group(1).strip() if support else "N/A")
                    with col3: st.metric("Recommended Next Step", next_step.group(1).strip() if next_step else "N/A")
                    
                    st.markdown("---")
                    st.subheader("Detailed Analysis")
                    st.markdown(result)
                    
                    # NEW: Save for presentation
                    st.session_state.current_analysis_summary = f"Hypothesis Test:\n{hypothesis}\n\nResult:\n{result}"
                    st.session_state.generated_presentation = None # Clear old presentation
                    
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Hypothesis Testing',
                        'hypothesis': hypothesis,
                        'data_context': data_context,
                        'result': result
                    })
                    st.success("Hypothesis testing completed!")
        else:
            st.warning("Please enter a hypothesis and provide context/data to test.")
    
    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 3: Multi-Perspective Analysis ---
elif tool_choice == "Multi-Perspective Analysis":
    st.header("🔭 Multi-Perspective Strategic Analysis")
    
    # Context building - automatically uses Global Context
    st.subheader("1. Analysis Context")
    context_input = st.text_area(
        "Describe the situation (Starts with Global Context):",
        height=150,
        value=st.session_state.global_context,
        placeholder="Describe your company, market situation, key challenges, strategic decisions needed..."
    )
    
    # Use uploaded documents as additional context (via Part objects)
    analysis_files = []
    if st.session_state.uploaded_data:
        st.subheader("Uploaded Documents Context")
        for filename, file_data in st.session_state.uploaded_data.items():
            if st.checkbox(f"Include {filename} in analysis", key=f"mpa_{filename}"):
                analysis_files.append(file_data['part'])
                context_input += f"\n\n--- INCLUDED DOCUMENT CONTEXT: {filename} ({file_data['type'].upper()}) ---"

    query_input = st.text_input(
        "What specific question or analysis do you need?",
        placeholder="e.g., Should we enter this new market? How can we improve profitability?"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Depth:",
            ["Quick Realist Analysis", "Comprehensive Multi-Perspective", "First Principles Breakdown"]
        )
    
    with col2:
        if analysis_type == "Comprehensive Multi-Perspective":
            generate_button = st.button("Generate All Perspectives", type="primary")
        else:
            generate_button = st.button("Generate Analysis", type="primary")
    
    if generate_button:
        if context_input and query_input:
            with st.spinner("Performing advanced strategic analysis..."):
                analysis = None
                all_perspectives = {}
                
                if analysis_type == "Comprehensive Multi-Perspective":
                    all_perspectives = generate_multiple_perspectives(context_input, query_input)
                    if all_perspectives:
                        st.subheader("Consolidated Strategic View")
                        tabs = st.tabs([p.capitalize() for p in all_perspectives.keys()])
                        
                        for tab, (perspective, analysis_text) in zip(tabs, all_perspectives.items()):
                            with tab:
                                st.subheader(f"{perspective.capitalize()} Perspective")
                                st.markdown(analysis_text)
                
                elif analysis_type == "First Principles Breakdown":
                    prompt = f"""
                    Perform a First Principles analysis on the following problem:
                    Context: {context_input}
                    Question: {query_input}
                    
                    Break it down to its fundamental, irreducible components and reconstruct innovative solutions.
                    """
                    analysis = get_gemini_response(prompt, 'gemini-2.5-pro') # Upgraded
                    if analysis:
                        st.subheader("First Principles Analysis")
                        st.markdown(analysis)
                
                else: # Quick Realist Analysis
                    analysis = get_advanced_analysis(context_input, query_input, "realist")
                    if analysis:
                        st.subheader("Strategic Analysis (Realist View)")
                        st.markdown(analysis)
            
                if analysis or all_perspectives:
                    response_content = analysis if analysis else str(all_perspectives)
                    
                    # NEW: Save for presentation
                    st.session_state.current_analysis_summary = f"Analysis Type: {analysis_type}\nContext: {context_input}\nQuery: {query_input}\n\nResult:\n{response_content}"
                    st.session_state.generated_presentation = None # Clear old presentation
                    
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Multi-Perspective Analysis',
                        'context': context_input,
                        'query': query_input,
                        'analysis_type': analysis_type,
                        'response': response_content
                    })
                    st.success("Analysis completed successfully!")
        else:
            st.warning("Please provide both context and a specific question.")

    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 4: Strategic Frameworks ---
elif tool_choice == "Strategic Frameworks":
    st.header("🔄 Advanced Strategic Frameworks")
    
    framework_choice = st.selectbox(
        "Choose Analytical Framework:",
        [
            "Second-Order Thinking",
            "Contrarian Analysis", 
            "Zero-Based Strategy",
            "Red Team Analysis",
            "Scenario Planning"
        ]
    )
    
    framework_input = st.text_area(
        f"Input for {framework_choice}: (Starts with Global Context)",
        height=200,
        value=st.session_state.global_context,
        placeholder="Describe the decision, strategy, or consensus view to analyze..."
    )
    
    if st.button(f"Apply {framework_choice}", type="primary"):
        if framework_input:
            with st.spinner(f"Applying {framework_choice}..."):
                result = None
                if framework_choice == "Second-Order Thinking":
                    prompt = f"Perform Second-Order Thinking analysis on this decision/strategy: {framework_input}\n\nMap out the immediate (first-order), secondary, and tertiary consequences. Structure with headings."
                elif framework_choice == "Contrarian Analysis":
                    prompt = f"Perform a Contrarian Analysis on this consensus view: {framework_input}\n\nWhat might the consensus be wrong about? What are the non-obvious, potentially profitable insights? Structure with headings."
                elif framework_choice == "Zero-Based Strategy":
                    prompt = f"Perform a Zero-Based Strategy review for this area: {framework_input}\n\nAssume a clean slate. Justify every activity and resource allocation from the ground up."
                else:
                    prompt = f"Perform {framework_choice} analysis on the following: {framework_input}\n\nProvide comprehensive insights and recommendations using the standard framework structure."
                
                result = get_gemini_response(prompt, 'gemini-2.5-pro') # Upgraded
                
                if result:
                    st.subheader(f"{framework_choice} Results")
                    st.markdown(result)
                    
                    # NEW: Save for presentation
                    st.session_state.current_analysis_summary = f"Framework: {framework_choice}\nInput: {framework_input}\n\nResult:\n{result}"
                    st.session_state.generated_presentation = None # Clear old presentation
                    
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Strategic Framework',
                        'framework': framework_choice,
                        'input': framework_input,
                        'output': result
                    })
                    st.success("Framework analysis completed!")
        else:
            st.warning("Please provide input for analysis.")

    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 5: Document Intelligence (Enhanced RAG Concept) ---
elif tool_choice == "Document Intelligence":
    st.header("📊 Document Intelligence (RAG)")
    
    doc_options = [name for name, data in st.session_state.uploaded_data.items() if data['type'] in ['pdf', 'txt']]
    if not doc_options:
        st.info("No PDF or TXT documents uploaded yet. Use the file uploader in the sidebar.")
    else:
        selected_doc = st.selectbox("Select Document for Analysis:", doc_options)
        
        if selected_doc:
            doc_content = st.session_state.uploaded_data[selected_doc]['content']
            doc_part = st.session_state.uploaded_data[selected_doc]['part']
            
            st.subheader("Document Preview")
            with st.expander(f"View Document Content ({len(doc_content):,} characters)"):
                st.text(doc_content[:1000] + "..." if len(doc_content) > 1000 else doc_content)
            
            st.subheader("Document Q&A")
            question = st.text_input("Ask a question about the document:")
            
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Direct Answer", "Critical Analysis", "Summary & Key Points", "Actionable Insights"]
            )
            
            if st.button("Analyze Document", type="primary"):
                if question:
                    with st.spinner("Analyzing document..."):
                        
                        if analysis_type == "Direct Answer":
                            prompt = f"Based on the attached document, answer directly and concisely: {question}"
                        elif analysis_type == "Critical Analysis":
                            prompt = f"Provide a critical analysis of the attached document content regarding: {question}"
                        elif analysis_type == "Summary & Key Points":
                            prompt = f"Summarize the key points from the attached document relevant to: {question}"
                        else: # Actionable Insights
                            prompt = f"Extract actionable insights and strategic next steps from the attached document regarding: {question}"
                        
                        # Pass the Part object as the files parameter
                        # Use Pro for complex RAG
                        response_text = get_gemini_response(prompt, 'gemini-2.5-pro', files=[doc_part]) 
                        
                        if response_text:
                            st.subheader(f"{analysis_type} Results")
                            st.markdown(response_text)
                            
                            # NEW: Save for presentation
                            st.session_state.current_analysis_summary = f"Document: {selected_doc}\nQuery: {question}\nAnalysis Type: {analysis_type}\n\nResult:\n{response_text}"
                            st.session_state.generated_presentation = None # Clear old presentation
                            
                            st.session_state.conversation_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'tool': 'Document Analysis',
                                'document': selected_doc,
                                'question': question,
                                'analysis_type': analysis_type,
                                'response': response_text
                            })
                            st.success("Document analysis completed!")
                else:
                    st.warning("Please enter a question about the document.")

    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 6: Data Analysis Suite (Enhanced AI Insights) ---
elif tool_choice == "Data Analysis Suite":
    st.header("📈 Data Analysis & Visualization")
    
    csv_options = [name for name, data in st.session_state.uploaded_data.items() if data['type'] == 'csv']
    if not csv_options:
        st.info("No CSV data files uploaded yet. Use the file uploader in the sidebar.")
    else:
        selected_csv = st.selectbox("Select Dataset:", csv_options)
        
        if selected_csv:
            df = st.session_state.uploaded_data[selected_csv]['content']
            data_part = st.session_state.uploaded_data[selected_csv]['part']
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Descriptive Statistics", "Correlation Analysis", "Trend Analysis", "Strategic Insights (AI)"]
            )
            
            if analysis_type == "Strategic Insights (AI)":
                custom_query = st.text_input("Specific query for AI insights:", placeholder="e.g., What are the drivers of high customer value?")

            if st.button("Perform Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    if analysis_type == "Descriptive Statistics":
                        st.subheader("Descriptive Statistics")
                        st.write(df.describe(include='all'))
                    
                    elif analysis_type == "Correlation Analysis":
                        st.subheader("Correlation Matrix (Numeric Data)")
                        numeric_df = df.select_dtypes(include=[np.number])
                        if not numeric_df.empty:
                            corr_matrix = numeric_df.corr()
                            fig = px.imshow(corr_matrix, aspect="auto", title="Correlation Matrix", text_auto=".2f")
                            st.plotly_chart(fig)
                        else:
                            st.warning("No numeric columns found for correlation analysis.")
                    
                    elif analysis_type == "Trend Analysis":
                        st.subheader("Data Trends (First 3 Numeric Columns)")
                        numeric_df = df.select_dtypes(include=[np.number])
                        if not numeric_df.empty:
                            try:
                                fig = px.line(numeric_df.iloc[:, :3].reset_index(), x='index', y=numeric_df.iloc[:, :3].columns, title="Trend Analysis Over Index")
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error plotting trend: {e}")
                        else:
                            st.warning("No numeric columns found for trend analysis.")
                    
                    elif analysis_type == "Strategic Insights (AI)":
                        data_summary = st.session_state.uploaded_data[selected_csv]['summary']
                        
                        prompt = f"""
                        Analyze the attached data summary and provide strategic business insights.
                        
                        **Project Context:** {st.session_state.global_context}
                        **Specific Query:** {custom_query if custom_query else 'Provide a general overview of key findings and recommendations.'}

                        Data Summary:
                        {data_summary}
                        
                        Provide a detailed analysis including key findings, potential correlations, and actionable business recommendations.
                        """
                        # Pass data Part object to AI (Pro for better data insights)
                        response = get_gemini_response(prompt, 'gemini-2.5-pro', files=[data_part])
                        if response:
                            st.subheader("Strategic Insights")
                            st.markdown(response)
                            
                            # NEW: Save for presentation
                            st.session_state.current_analysis_summary = f"Data Analysis: {selected_csv}\nQuery: {custom_query}\n\nResult:\n{response}"
                            st.session_state.generated_presentation = None # Clear old presentation
                            
                            st.session_state.conversation_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'tool': 'Data Analysis',
                                'dataset': selected_csv,
                                'analysis_type': analysis_type,
                                'response': response
                            })
                            st.success("Data analysis completed!")
                            
    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 7: Strategic Prioritization (Enhanced with Structured Output) ---
elif tool_choice == "Strategic Prioritization":
    st.header("🎯 Strategic Prioritization (Impact vs. Effort)")
    st.markdown("**Prioritize initiatives and recommendations based on AI-assessed impact and effort.**")
    
    recommendations_input = st.text_area(
        "Enter your strategic recommendations (one per line or separated by commas):",
        height=150,
        placeholder="""Launch new mobile app to reach younger demographics
Implement AI-powered customer service chatbot 
Expand to European markets"""
    )
    
    if st.button("Prioritize Recommendations", type="primary"):
        if recommendations_input:
            with st.spinner("Analyzing and prioritizing recommendations..."):
                prioritization_output_dict = prioritize_recommendations(recommendations_input)
                
                if prioritization_output_dict:
                    
                    # The response is now a dict guaranteed by the Pydantic schema
                    recommendations_data = prioritization_output_dict.get('recommendations_data', [])
                    textual_analysis = prioritization_output_dict.get('textual_analysis', 'No textual analysis provided by AI.')
                    
                    # 2. Update Session State
                    st.session_state.prioritization_matrix = {
                        'recommendations': recommendations_input,
                        'analysis': textual_analysis,
                        'data': recommendations_data, 
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📈 Prioritization Analysis")
                        st.markdown(textual_analysis)
                    
                    with col2:
                        st.subheader("📊 Impact vs Effort Matrix")
                        
                        # 3. Use dynamic data for plotting
                        fig = create_prioritization_plot(recommendations_data) 
                        if fig:
                            st.plotly_chart(fig)
                        else:
                            st.info("Prioritization data is unavailable or could not be parsed for plotting.")
                    
                    # NEW: Save for presentation
                    st.session_state.current_analysis_summary = f"Prioritization Analysis:\n{textual_analysis}\n\nData:\n{json.dumps(recommendations_data, indent=2)}"
                    st.session_state.generated_presentation = None # Clear old presentation
                    
                    # Store in history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Strategic Prioritization',
                        'recommendations': recommendations_input,
                        'analysis': textual_analysis,
                        'data': recommendations_data
                    })
                    st.success("Prioritization completed!")
        else:
            st.warning("Please enter recommendations to prioritize.")

    # NEW: Display presentation generator
    if st.session_state.current_analysis_summary:
        display_presentation_output()


# --- TOOL 8: Live Chat Consultation ---
elif tool_choice == "Live Chat Consultation":
    st.header("💬 Live Chat Consultation")
    st.markdown("**Your personalized strategic consultant for real-time questions.**")
    
    if st.session_state.current_chat is None:
        initialize_chat()
        st.success("New consultation chat started!")
    
    if st.session_state.current_chat:
        try:
            history_messages = st.session_state.current_chat.get_history()
            for message in history_messages:
                if message.parts[0].text.strip() == "": continue

                if message.role == "user":
                    st.chat_message("user").write(message.parts[0].text)
                else:
                    st.chat_message("assistant").write(message.parts[0].text)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
    
    user_input = st.chat_input("Type your consulting question here...")
    
    if user_input:
        st.chat_message("user").write(user_input)
        
        with st.spinner("Consultant is thinking..."):
            response = send_chat_message(user_input)
            if response:
                st.chat_message("assistant").write(response)
        
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat Session"):
            initialize_chat()
            st.rerun()
    with col2:
        if st.button("Save Chat to History"):
            if st.session_state.current_chat:
                try:
                    chat_history = st.session_state.current_chat.get_history()
                    chat_content = "\n".join([f"{msg.role.capitalize()}: {msg.parts[0].text}" for msg in chat_history if msg.parts[0].text])
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'tool': 'Live Chat Session',
                        'content': chat_content
                    })
                    st.success("Chat saved to history!")
                except Exception as e:
                    st.error(f"Error saving chat: {e}")


# --- TOOL 9: Conversation History ---
elif tool_choice == "Conversation History":
    st.header("💬 Analysis History")
    
    if st.session_state.conversation_history:
        st.info(f"Total analyses: {len(st.session_state.conversation_history)}")
        
        for i, conversation in enumerate(reversed(st.session_state.conversation_history[-20:])):
            timestamp = conversation.get('timestamp', '')[:19].replace('T', ' ')
            tool_name = conversation.get('tool', 'Unknown Tool')
            
            with st.expander(f"Analysis {len(st.session_state.conversation_history)-i} - **{tool_name}** ({timestamp})", expanded=i==0):
                
                st.write(f"**Tool:** {tool_name}")
                
                if tool_name == 'Live Chat Session':
                    st.text_area("Chat Content", conversation['content'], height=200, key=f"chat_{i}")
                else:
                    if 'problem' in conversation: st.write(f"**Problem:** {conversation['problem']}")
                    if 'query' in conversation: st.write(f"**Query:** {conversation.get('query', conversation.get('question'))}")
                    if 'framework' in conversation: st.write(f"**Framework:** {conversation['framework']}")
                    if 'hypothesis' in conversation: st.write(f"**Hypothesis:** {conversation['hypothesis']}")
                        
                    response_text = conversation.get('response', conversation.get('analysis', conversation.get('structure', 'No detailed response saved.')))
                    
                    if isinstance(response_text, str):
                        st.markdown("**Summary/Analysis:**")
                        st.markdown(response_text)
                    else:
                        st.write("**Full Response (Structured Data):**")
                        st.json(response_text)
    else:
        st.info("No analysis history yet. Start using the tools to build your history!")


# --- Advanced Features Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Actions")

if st.sidebar.button("Clear All Project Data", type="secondary"):
    for key in list(st.session_state.keys()):
        if key not in ['api_key']: # Preserve API key
            del st.session_state[key]
    st.rerun()

# Export functionality
if st.session_state.conversation_history:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategic_analysis_{timestamp}.json"
    
    # Prepare data for export
    json_string = json.dumps(st.session_state.conversation_history, indent=2, default=str)
    
    st.sidebar.download_button(
        label="📥 Export Analysis History",
        data=json_string,
        file_name=filename,
        mime="application/json",
        use_container_width=True
    )

# --- Real-time Context Display ---
st.sidebar.markdown("---")
st.sidebar.subheader("Current Project Status")
if st.session_state.global_context:
    st.sidebar.write("🌎 **Global Context Active**")
if st.session_state.uploaded_data:
    st.sidebar.write(f"📁 Documents/Data: **{len(st.session_state.uploaded_data)}**")
if st.session_state.conversation_history:
    st.sidebar.write(f"💬 Analyses Saved: **{len(st.session_state.conversation_history)}**")

# --- Footer ---
st.markdown("---")
st.markdown("""
### 🎯 Next Steps: Engage the AI Consultant

*Select a tool from the sidebar to begin the project lifecycle:*
* **Problem Structuring:** Define the problem space.
* **Hypothesis Testing:** Validate assumptions with data.
* **Deep Analysis:** Get multi-perspective strategic insights.
""")
