import streamlit as st
import google.generativeai as genai
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


# Load environment variables (for local development)
load_dotenv()

# Configure the Gemini API - UPDATED FOR STREAMLIT SECRETS
# Try to get API key from Streamlit secrets first, then from environment variables
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
    st.success("✅ API key loaded from Streamlit secrets")
else:
    # Fallback to environment variable for local development
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.success("✅ API key loaded from environment variable")

if not api_key:
    st.error("""
    ❌ GEMINI_API_KEY not found. 
    
    Please add your Gemini API key to Streamlit Secrets:
    1. Go to your app dashboard
    2. Click on "Settings" ⚙️
    3. Go to "Secrets" tab
    4. Add: `GEMINI_API_KEY = "your_actual_api_key_here"`
    
    For local development, create a .env file with: GEMINI_API_KEY=your_key
    """)
    st.stop()

try:
    # Configure the API - using the standard approach
    genai.configure(api_key=api_key)
    st.success("✅ Gemini API configured successfully!")
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
def get_gemini_response(prompt_text, model_name='gemini-pro', response_schema=None, files=None):
    """A robust helper function to call the Gemini API with optional files and structured output."""
    try:
        # For older versions, use the standard generate_content approach
        model = genai.GenerativeModel(model_name)
        
        # Prepare contents
        contents = [prompt_text]
        
        response = model.generate_content(contents)
        
        # For structured output, we'll need to parse the response text
        if response_schema:
            try:
                # Try to parse JSON from the response
                return json.loads(response.text)
            except json.JSONDecodeError:
                # If it's not JSON, return the text as is
                return {"text": response.text}
        
        return response.text
        
    except Exception as e:
        st.error(f"Gemini API Error ({model_name}): {e}")
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
    
    with st.spinner("Generating bespoke presentation code with Gemini..."):
        # Use the powerful model for this creative task
        full_response = get_gemini_response(final_prompt, 'gemini-pro') 
        
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


# --- Chat Functionality (simplified for compatibility) ---
def initialize_chat():
    """Initialize a new chat session"""
    try:
        # Use standard model for chat
        model = genai.GenerativeModel('gemini-pro')
        st.session_state.current_chat = model.start_chat(history=[])
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
    return get_gemini_response(prompt, 'gemini-pro')

def generate_hypotheses(problem_statement):
    """Generate a list of testable business hypotheses using structured output."""
    prompt = f"""
    You are a strategic consultant. Based on this core problem, generate 5-7 distinct, testable, and mutually exclusive hypotheses that could explain the problem or suggest a solution path.
    
    CORE PROBLEM: {problem_statement}
    
    Return the output as a valid JSON object with a key "hypotheses" containing the list of hypotheses.
    """
    response = get_gemini_response(prompt, 'gemini-pro')
    if response:
        # Try to extract JSON from response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON found, create a simple structure
                lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith(('#', '-', '*'))]
                hypotheses = [line for line in lines if len(line) > 10][:7]  # Take meaningful lines as hypotheses
                return {"hypotheses": hypotheses}
        except:
            # Fallback: return the text as hypotheses
            return {"hypotheses": [response]}
    return None

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
    return get_gemini_response(prompt, 'gemini-pro')

def prioritize_recommendations(recommendations):
    """Create a prioritization matrix for strategic recommendations using Pydantic schema."""
    prompt = f"""
    You are a strategic prioritization expert. Analyze these recommendations, which are separated by newlines or commas. For each recommendation, assess its **Potential Impact** (High/Medium/Low) and **Implementation Effort** (High/Medium/Low).
    
    Recommendations:
    {recommendations}
    
    Provide your analysis in JSON format with two keys: 
    - "recommendations_data": a list of objects, each with "recommendation", "impact", and "effort" fields
    - "textual_analysis": a comprehensive textual analysis categorizing results into Quick Wins, Major Projects, Fill-ins, and Avoid with clear reasoning.
    """
    response = get_gemini_response(prompt, 'gemini-pro')
    if response:
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: create a simple structure
                lines = [line.strip() for line in recommendations.split('\n') if line.strip()]
                recommendations_data = []
                for i, line in enumerate(lines):
                    recommendations_data.append({
                        "recommendation": line,
                        "impact": "Medium",
                        "effort": "Medium"
                    })
                return {
                    "recommendations_data": recommendations_data,
                    "textual_analysis": response
                }
        except:
            return None
    return None


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

    # Ensure recommendations_data is a list of dictionaries
    for rec in recommendations_data:
        rec_dict = rec if isinstance(rec, dict) else rec
        
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
    return get_gemini_response(prompt, 'gemini-pro')

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


# --- File Processing Functions (Simplified for compatibility) ---

def extract_pdf_text_and_part(uploaded_file):
    """Extracts text for context."""
    text = ""
    try:
        # Reset file pointer to the beginning for PyPDF2
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        
        return text, None  # Return None for part in simplified version
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None, None

def process_csv_data(uploaded_file):
    """Process CSV data and return DataFrame, summary."""
    try:
        # Reset file pointer to the beginning for pandas
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # Create a text summary for the AI to ingest
        data_summary = f"Dataset columns: {list(df.columns)}\nData Head:\n{df.head().to_string()}"
        
        return df, None, data_summary  # Return None for part in simplified version
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
**Professional consulting frameworks powered by advanced AI analysis (Gemini with Structured Output)**
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
tool_choice = st.sidebar.radio("Select Tool:", tool_choice_keys, key="tool_choice")

# Get the actual tool name for the main logic
actual_tool = tool_map[tool_group][tool_choice]

# --- Main Content Area ---
if actual_tool == "Problem Structuring Workbench":
    st.header("🌳 Problem Structuring Workbench")
    st.markdown("Use MECE principles to break down complex problems into manageable components.")
    
    problem = st.text_area("Enter the core problem statement:", height=100, 
                          placeholder="e.g., 'Our e-commerce conversion rate has declined by 15% in the last quarter despite increased traffic.'")
    
    if st.button("Build Issue Tree", use_container_width=True):
        if problem:
            with st.spinner("Building comprehensive issue tree..."):
                issue_tree = build_issue_tree(problem)
                if issue_tree:
                    st.markdown("### 🎯 Structured Issue Tree")
                    st.markdown(issue_tree)
                    # Store for context and potential presentation
                    st.session_state.current_analysis_summary = f"Problem: {problem}\n\nIssue Tree Analysis:\n{issue_tree}"
                else:
                    st.error("Failed to generate issue tree. Please try again.")
        else:
            st.warning("Please enter a problem statement.")

elif actual_tool == "Hypothesis Testing":
    st.header("🔬 Hypothesis Testing & Generation")
    st.markdown("Generate and test business hypotheses systematically.")
    
    tab1, tab2 = st.tabs(["Generate Hypotheses", "Test Hypotheses"])
    
    with tab1:
        st.subheader("Generate Testable Hypotheses")
        problem_for_hypotheses = st.text_area("Problem for hypothesis generation:", height=80,
                                             placeholder="e.g., 'Why are customer churn rates increasing?'")
        
        if st.button("Generate Hypotheses", key="gen_hypotheses"):
            if problem_for_hypotheses:
                with st.spinner("Generating testable hypotheses..."):
                    hypotheses_result = generate_hypotheses(problem_for_hypotheses)
                    if hypotheses_result and 'hypotheses' in hypotheses_result:
                        st.markdown("### 🎯 Generated Hypotheses")
                        for i, hypothesis in enumerate(hypotheses_result['hypotheses'], 1):
                            st.markdown(f"**Hypothesis {i}:** {hypothesis}")
                        
                        # Store for context
                        st.session_state.current_analysis_summary = f"Problem: {problem_for_hypotheses}\n\nGenerated Hypotheses:\n" + "\n".join([f"{i}. {h}" for i, h in enumerate(hypotheses_result['hypotheses'], 1)])
                    else:
                        st.error("Failed to generate hypotheses. Please try again.")
            else:
                st.warning("Please enter a problem statement.")
    
    with tab2:
        st.subheader("Test Existing Hypotheses")
        hypothesis_to_test = st.text_area("Hypothesis to test:", height=60,
                                         placeholder="e.g., 'Customers are churning due to poor onboarding experience'")
        
        data_context = st.text_area("Available data/context for testing:", height=100,
                                   placeholder="e.g., 'Survey data shows 40% of churned customers mentioned onboarding. Support tickets show increased onboarding questions.'")
        
        if st.button("Test Hypothesis", key="test_hypothesis"):
            if hypothesis_to_test and data_context:
                with st.spinner("Testing hypothesis against available data..."):
                    test_result = test_hypothesis(hypothesis_to_test, data_context)
                    if test_result:
                        st.markdown("### 📊 Hypothesis Test Results")
                        st.markdown(test_result)
                        
                        # Store for context
                        st.session_state.current_analysis_summary = f"Hypothesis: {hypothesis_to_test}\n\nData Context: {data_context}\n\nTest Results:\n{test_result}"
                    else:
                        st.error("Failed to test hypothesis. Please try again.")
            else:
                st.warning("Please provide both a hypothesis and data context.")

elif actual_tool == "Multi-Perspective Analysis":
    st.header("🎭 Multi-Perspective Analysis")
    st.markdown("Analyze problems from multiple thinking frameworks to avoid blind spots.")
    
    analysis_context = st.text_area("Context for analysis:", height=120,
                                   placeholder="e.g., 'Company is considering entering a new market in Southeast Asia. Current revenue: $50M. New market potential: $20M in 3 years.'")
    
    analysis_query = st.text_input("Specific question or analysis focus:",
                                  placeholder="e.g., 'What are the risks and opportunities of this market entry?'")
    
    if st.button("Generate Multi-Perspective Analysis", use_container_width=True):
        if analysis_context and analysis_query:
            with st.spinner("Generating comprehensive multi-perspective analysis..."):
                perspectives = generate_multiple_perspectives(analysis_context, analysis_query)
                
                if perspectives:
                    st.markdown("### 📋 Multi-Perspective Analysis Results")
                    
                    # Create tabs for each perspective
                    tabs = st.tabs([f"🧠 {p.capitalize()}" for p in perspectives.keys()])
                    
                    all_analysis = ""
                    for (perspective, analysis), tab in zip(perspectives.items(), tabs):
                        with tab:
                            st.markdown(analysis)
                            all_analysis += f"**{perspective.upper()} PERSPECTIVE:**\n{analysis}\n\n"
                    
                    # Store comprehensive analysis for presentation
                    st.session_state.current_analysis_summary = f"Context: {analysis_context}\n\nQuery: {analysis_query}\n\n{all_analysis}"
                else:
                    st.error("Failed to generate analysis. Please try again.")
        else:
            st.warning("Please provide both context and a specific query.")

elif actual_tool == "Strategic Prioritization":
    st.header("🎯 Strategic Prioritization Matrix")
    st.markdown("Evaluate and prioritize strategic initiatives based on impact vs. effort.")
    
    recommendations_input = st.text_area("Enter recommendations (one per line or comma-separated):", height=120,
                                        placeholder="e.g., 'Launch new mobile app\nImplement AI chatbot\nOptimize website speed\nExpand to European markets'")
    
    if st.button("Create Prioritization Matrix", use_container_width=True):
        if recommendations_input:
            with st.spinner("Analyzing and prioritizing recommendations..."):
                prioritization_result = prioritize_recommendations(recommendations_input)
                
                if prioritization_result and 'recommendations_data' in prioritization_result:
                    # Display the visualization
                    st.markdown("### 📊 Prioritization Matrix")
                    fig = create_prioritization_plot(prioritization_result['recommendations_data'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display the detailed analysis
                    if 'textual_analysis' in prioritization_result:
                        st.markdown("### 📝 Strategic Analysis")
                        st.markdown(prioritization_result['textual_analysis'])
                    
                    # Store for context and presentation
                    recommendations_text = "\n".join([f"- {r['recommendation']} (Impact: {r['impact']}, Effort: {r['effort']})" 
                                                     for r in prioritization_result['recommendations_data']])
                    st.session_state.current_analysis_summary = f"Recommendations Prioritization:\n{recommendations_text}\n\nAnalysis:\n{prioritization_result.get('textual_analysis', '')}"
                else:
                    st.error("Failed to generate prioritization matrix. Please try again.")
        else:
            st.warning("Please enter some recommendations to prioritize.")

elif actual_tool == "Live Chat Consultation":
    st.header("💬 Live Chat Consultation")
    st.markdown("Real-time conversation with the AI consultant for open-ended discussion.")
    
    # Initialize chat if not already done
    if st.session_state.current_chat is None:
        initialize_chat()
    
    # Display chat history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your strategic question..."):
        # Add user message to chat
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt)
                if response:
                    st.markdown(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    
                    # Store the conversation for context
                    st.session_state.current_analysis_summary = f"Live Chat Consultation Summary:\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history[-6:]])  # Last 6 messages
                else:
                    st.error("Failed to get response. Please try again.")

elif actual_tool == "Conversation History":
    st.header("📋 Conversation History")
    st.markdown("Review your consultation session history.")
    
    if st.session_state.conversation_history:
        for i, message in enumerate(st.session_state.conversation_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    else:
        st.info("No conversation history yet. Start a chat in the Live Consultation tab!")

elif actual_tool == "Document Intelligence":
    st.header("📄 Document Intelligence (RAG)")
    st.markdown("Upload documents to extract insights and enable document-aware analysis.")
    
    uploaded_file = st.file_uploader("Upload PDF or CSV", type=['pdf', 'csv'])
    
    if uploaded_file:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        if file_type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                text, _ = extract_pdf_text_and_part(uploaded_file)
                if text:
                    st.success(f"✅ Extracted {len(text)} characters from {file_name}")
                    st.session_state.uploaded_data[file_name] = text
                    
                    # Show preview
                    with st.expander("📖 Document Preview"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
        
        elif file_type == "text/csv":
            with st.spinner("Processing CSV data..."):
                df, _, summary = process_csv_data(uploaded_file)
                if df is not None:
                    st.success(f"✅ Processed {len(df)} rows from {file_name}")
                    st.session_state.uploaded_data[file_name] = df
                    
                    # Show preview
                    with st.expander("📊 Data Preview"):
                        st.dataframe(df.head())
                        st.markdown(f"**Data Summary:** {summary}")

elif actual_tool == "Data Analysis Suite":
    st.header("📈 Data Analysis Suite")
    st.markdown("Advanced data analysis and visualization tools.")
    
    if st.session_state.uploaded_data:
        st.success(f"✅ Loaded {len(st.session_state.uploaded_data)} data source(s)")
        
        # Simple analysis example
        for name, data in st.session_state.uploaded_data.items():
            if isinstance(data, pd.DataFrame):
                st.subheader(f"Analysis: {name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Statistics**")
                    st.dataframe(data.describe())
                
                with col2:
                    st.markdown("**Column Information**")
                    col_info = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes,
                        'Non-Null': data.count(),
                        'Null': data.isnull().sum()
                    })
                    st.dataframe(col_info)
    else:
        st.info("📁 Upload documents in the Document Intelligence tab to enable data analysis.")

elif actual_tool == "Strategic Frameworks":
    st.header("🏛️ Strategic Frameworks")
    st.markdown("Apply classic strategic frameworks to your business challenges.")
    
    framework = st.selectbox(
        "Choose a strategic framework:",
        ["SWOT Analysis", "Porter's Five Forces", "PESTLE Analysis", "BCG Matrix", "Ansoff Matrix"]
    )
    
    business_context = st.text_area("Business context for framework application:", height=120,
                                   placeholder="e.g., 'Tech startup in SaaS space, 3 years old, $5M ARR, facing increased competition...'")
    
    if st.button(f"Apply {framework}", use_container_width=True):
        if business_context:
            with st.spinner(f"Applying {framework} to your business context..."):
                prompt = f"""
                You are a strategic consultant. Apply the {framework} framework to this business context:
                
                {business_context}
                
                Provide a comprehensive, structured analysis using the {framework} framework.
                Include specific, actionable insights and recommendations.
                """
                
                analysis = get_gemini_response(prompt, 'gemini-pro')
                if analysis:
                    st.markdown(f"### 🎯 {framework} Analysis")
                    st.markdown(analysis)
                    
                    # Store for context
                    st.session_state.current_analysis_summary = f"Framework: {framework}\n\nContext: {business_context}\n\nAnalysis:\n{analysis}"
                else:
                    st.error(f"Failed to generate {framework} analysis. Please try again.")
        else:
            st.warning("Please provide business context.")

# --- NEW: Always show presentation generator if there's analysis content ---
if st.session_state.current_analysis_summary:
    display_presentation_output()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Advanced Strategic Consulting AI | Powered by Google Gemini | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
