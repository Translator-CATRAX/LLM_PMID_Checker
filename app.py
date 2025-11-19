#!/usr/bin/env python3
"""Streamlit web interface for LLM PMID Checker."""
import streamlit as st
import asyncio
import logging
import pandas as pd
import plotly.express as px
import requests
from typing import List, Optional
from datetime import datetime
import json

# Import project modules
from src.triple_evaluator import TripleEvaluatorSystem, TripleEvaluationResult
from src.node_normalization import NodeNormalizationClient
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_METADATA = {
    # Ollama Models
    "hermes4:70b-iq4-xs": {
        "label": "Hermes 4 70B (IQ4_XS)",
        "type": "ollama",
        "details": [
            "Fastest Hermes 4 quantization",
            "Lowest VRAM usage (~28 GB)",
            "Best when GPU memory is constrained"
        ],
    },
    "hermes4:70b-q4-s": {
        "label": "Hermes 4 70B (Q4_S)",
        "type": "ollama",
        "details": [
            "Balanced speed and quality",
            "Requires ~34 GB VRAM",
            "Good trade-off for most workloads"
        ],
    },
    "hermes4:70b-q4-m": {
        "label": "Hermes 4 70B (Q4_M)",
        "type": "ollama",
        "details": [
            "Highest quality Hermes 4 quantization",
            "Requires ~42 GB VRAM",
            "Best accuracy for critical reviews"
        ],
    },
    "gpt-oss:20b": {
        "label": "GPT-OSS 20B",
        "type": "ollama",
        "details": [
            "20B parameter open-source mixture",
            "~12 GB VRAM",
            "Strong baseline and faster than Hermes 4"
        ],
    },
    # OpenAI Models
    "gpt-5-nano": {
        "label": "GPT-5 Nano",
        "type": "openai",
        "details": [
            "OpenAI's smallest GPT-5 model",
            "Cloud-based (requires API key)",
            "Fast and cost-effective"
        ],
    },
    "gpt-5-mini": {
        "label": "GPT-5 Mini",
        "type": "openai",
        "details": [
            "OpenAI's mid-tier GPT-5 model",
            "Cloud-based (requires API key)",
            "Balanced performance and cost"
        ],
    },
}


def get_model_label(model_name: str) -> str:
    """Return a user-friendly label for a model."""
    return MODEL_METADATA.get(model_name, {}).get("label", model_name)


def show_model_info(model_name: str):
    """Render contextual information for the selected model."""
    metadata = MODEL_METADATA.get(model_name)
    if not metadata:
        st.info(f"🧠 **{model_name}**")
        return

    details = metadata.get("details", [])
    
    # Add web search warning for OpenAI models if enabled
    if metadata.get("type") == "openai" and settings.openai_enable_web_search:
        details = details + ["⚠️ Web search enabled (+$10-50/1k searches)"]
    
    detail_text = "\n".join(f"- {line}" for line in details) if details else ""
    st.info(f"🧠 **{metadata['label']}**\n{detail_text}")

# Page configuration
st.set_page_config(
    page_title="LLM PMID Checker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check if the environment is properly configured."""
    missing_vars = []
    warnings = []
    
    # Check required environment variables
    if not settings.ncbi_email:
        missing_vars.append("NCBI_EMAIL")
    if not settings.ncbi_api_key:
        warnings.append("NCBI_API_KEY (optional but recommended for higher rate limits)")
    
    # Check Ollama connection
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/version", timeout=5)
        ollama_status = "✅ Connected" if response.status_code == 200 else "❌ Error"
    except Exception as e:
        ollama_status = f"❌ Not accessible ({str(e)[:50]}...)"
    
    # Check OpenAI API key
    if not settings.openai_api_key:
        warnings.append("OPENAI_API_KEY (required for OpenAI models)")
        openai_status = "⚠️ Not configured"
    else:
        openai_status = "✅ API key set"
    
    return {
        "missing_vars": missing_vars,
        "warnings": warnings,
        "ollama_status": ollama_status,
        "openai_status": openai_status
    }

def display_environment_status():
    """Display environment configuration status."""
    status = check_environment()
    
    st.sidebar.header("🔧 Environment Status")
    
    # Ollama status
    st.sidebar.write("**Ollama Models**")
    st.sidebar.write(f"• Server: {status['ollama_status']}")
    st.sidebar.write(f"• Base URL: {settings.ollama_base_url}")
    
    # OpenAI status
    st.sidebar.write("**OpenAI Models**")
    st.sidebar.write(f"• API: {status['openai_status']}")
    if settings.openai_enable_web_search:
        st.sidebar.warning("⚠️ Web search enabled (+$10-50/1k calls)")
    
    st.sidebar.divider()
    
    # Missing variables
    if status["missing_vars"]:
        st.sidebar.error(f"❌ Missing: {', '.join(status['missing_vars'])}")
    else:
        st.sidebar.success("✅ Required variables set")
    
    # Warnings
    if status["warnings"]:
        for warning in status["warnings"]:
            st.sidebar.warning(f"⚠️ {warning}")
    

async def run_evaluation(
    triple_data: dict,
    pmids: List[str],
    model: str,
    checker_model: Optional[str] = None,
    progress_callback=None
) -> TripleEvaluationResult:
    """Run the triple check asynchronously with progress tracking."""
    try:
        if progress_callback:
            progress_callback(10, "🔧 Initializing checking system...")
        
        # Use checker_model for verification (None means disable)
        checker = checker_model if checker_model and checker_model.lower() != 'none' else None
        evaluator = TripleEvaluatorSystem(llm_provider=model, checker_model=checker)
        
        if progress_callback:
            progress_callback(20, "📡 Fetching abstracts from PubMed...")
        
        # Get abstracts first
        abstract_data = evaluator.pmid_extractor.extract_abstracts(pmids)
        
        # Entity normalization is already done in the UI, so skip that step
        if progress_callback:
            progress_callback(40, "🤖 Starting AI model evaluation...")
        
        # Manual evaluation with progress tracking for each PMID
        from src.evaluation_agent import TripleData, TripleEvaluation
        from src.triple_evaluator import TripleEvaluationResult
        
        # Create enriched triple data
        triple_obj = TripleData(
            subject=triple_data['subject'],
            predicate=triple_data['predicate'],
            object=triple_data['object'],
            subject_names=triple_data.get('subject_names'),
            object_names=triple_data.get('object_names'),
            qualified_predicate=triple_data.get('qualified_predicate'),
            qualified_object_aspect=triple_data.get('qualified_object_aspect'),
            qualified_object_direction=triple_data.get('qualified_object_direction')
        )
        
        evaluations = []
        
        # Separate valid and invalid PMIDs
        valid_pmids = []
        for pmid in pmids:
            data = abstract_data.get(pmid)
            if not data:
                # PMID not found in results
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning="PMID not found in results",
                    subject_mentioned=False,
                    object_mentioned=False
                ))
                continue
                
            if data.error:
                # Error extracting abstract
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning=f"Error: {data.error}",
                    subject_mentioned=False,
                    object_mentioned=False
                ))
                continue
            
            if not data.abstract.strip():
                # No abstract available
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning="No abstract available",
                    subject_mentioned=False,
                    object_mentioned=False
                ))
                continue
            
            # Add to valid PMIDs list
            valid_pmids.append((pmid, data.title, data.abstract))
        
        # Process valid PMIDs with concurrent batch processing
        if valid_pmids:
            if progress_callback:
                progress_callback(40, f"🔬 Starting batch evaluation of {len(valid_pmids)} PMIDs...")
            
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
            
            async def evaluate_with_progress(idx: int, pmid: str, title: str, abstract: str) -> TripleEvaluation:
                """Evaluate a single PMID with progress tracking."""
                async with semaphore:
                    try:
                        if progress_callback:
                            current_progress = 40 + int((idx / len(valid_pmids)) * 50)
                            progress_callback(current_progress, f"🔬 Evaluating PMID {pmid} ({idx+1}/{len(valid_pmids)})...")
            
                evaluation = await evaluator.evaluation_agent.evaluate_triple_against_abstract(
                    triple=triple_obj,
                            abstract=abstract,
                    pmid=pmid,
                            title=title
                )
                
                # Apply validation rules
                evaluation = evaluator._validate_evaluation_logic(evaluation, pmid)
                        return evaluation
                
            except Exception as e:
                        return TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning=f"Evaluation failed: {str(e)}",
                    subject_mentioned=False,
                    object_mentioned=False
                        )
            
            # Create tasks for all valid PMIDs
            tasks = [
                evaluate_with_progress(i, pmid, title, abstract)
                for i, (pmid, title, abstract) in enumerate(valid_pmids)
            ]
            
            # Execute all tasks concurrently
            batch_evaluations = await asyncio.gather(*tasks)
            evaluations.extend(batch_evaluations)
        
        if progress_callback:
            progress_callback(90, "📊 Finalizing results...")
        
        # Sort evaluations by original PMID order
        pmid_order = {pmid: idx for idx, pmid in enumerate(pmids)}
        evaluations.sort(key=lambda x: pmid_order.get(x.pmid, float('inf')))
        
        result = TripleEvaluationResult(triple=triple_obj, evaluations=evaluations)
        
        
        if progress_callback:
            progress_callback(100, "✅ Evaluation completed!")
        
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise e

def create_results_dataframe(result: TripleEvaluationResult) -> pd.DataFrame:
    """Create a pandas DataFrame from evaluation results."""
    data = []
    for eval_result in result.evaluations:
        # Map evidence categories to display formats
        category_map = {
            "direct_support": "Direct support",
            "opposite_assertion": "Opposite Assertion",
            "missing_qualifier": "Missing qualifier",
            "wrong_qualifier": "Wrong qualifier",
            "not_supported": "Not supported"
        }
        category_display = category_map.get(eval_result.evidence_category, "Unknown")
        
        # Only direct_support is considered supported
        is_supported = eval_result.evidence_category == "direct_support"
        
        # Reorder columns: PMID, Supported, Short Explanation, Subject Mentioned, Object Mentioned, Supporting Sentence, Reasoning
        data.append({
            'PMID': f"https://pubmed.ncbi.nlm.nih.gov/{eval_result.pmid}",  # Full URL for linking
            'Supported': is_supported,
            'Category': category_display,
            'Subject Mentioned': "Yes" if eval_result.subject_mentioned else "No",
            'Object Mentioned': "Yes" if eval_result.object_mentioned else "No",
            'Supporting Sentence': eval_result.supporting_sentence or '',
            'Reasoning': eval_result.reasoning or ''
        })
    return pd.DataFrame(data)

def create_summary_charts(df: pd.DataFrame):
    """Create summary visualizations."""
    # Pie chart of support distribution
    support_counts = df['Supported'].value_counts()
    
    fig_pie = px.pie(
        values=support_counts.values,
        names=['Supported' if x else 'Not Supported' for x in support_counts.index],
        title="Support Distribution",
        color_discrete_map={
            'Supported': '#2E8B57',
            'Not Supported': '#DC143C'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)



def run_app():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
    <div style="padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">🧬 LLM PMID Checker</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; text-align: center;">
            Check research triples against PubMed abstracts using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check environment and display status
    display_environment_status()
    
    # Main content
    main_container()

def main_container():
    """Main application container."""
    
    # Triple input section
    st.header("🔬 Research Triple")
    st.markdown("Enter the research relationship you want to check:")
    
    # Input mode selection
    input_mode = st.radio(
        "Choose input format:",
        ["Entity Names (with normalization)", "CURIEs (direct identifiers)"],
        help="Entity Names: Automatically finds equivalent names (e.g., 'SIX1'). CURIEs: Direct semantic identifiers (e.g., 'NCBIGene:6495')",
        horizontal=True
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if input_mode == "Entity Names (with normalization)":
            subject = st.text_input("Subject", value="SIX1", help="Gene/protein name (will be normalized)", key="entity_subject").strip()
        else:
            subject = st.text_input("Subject CURIE", value="NCBIGene:6495", help="Subject CURIE (e.g., NCBIGene:6495)", key="curie_subject").strip()
    with col2:
        predicate = st.text_input("Predicate", value="affects", help="The relationship or action", key="predicate_input").strip()
    with col3:
        if input_mode == "Entity Names (with normalization)":
            object_ = st.text_input("Object", value="Cell Proliferation", help="Process/condition name (will be normalized)", key="entity_object").strip()
        else:
            object_ = st.text_input("Object CURIE", value="UMLS:C0596290", help="Object CURIE (e.g., UMLS:C0596290)", key="curie_object").strip()
    
    # Qualifier inputs section
    st.subheader("🔍 Optional Qualifiers")
    st.markdown("Add qualifiers to make the relationship more specific")
    
    with st.expander("Add Qualifiers", expanded=False):
        qualifier_col1, qualifier_col2, qualifier_col3 = st.columns(3)
        with qualifier_col1:
            qualified_predicate = st.text_input("Qualified Predicate", 
                                               value="", 
                                               help="More specific relationship (e.g., 'causes', 'regulates'). Required if using any qualifiers.",
                                               key="new_qualified_predicate").strip()
        with qualifier_col2:
            qualified_object_aspect = st.text_input("Object Aspect", 
                                                   value="", 
                                                   help="Aspect of the object (e.g., 'activity', 'abundance', 'activity_or_abundance')",
                                                   key="new_qualified_object_aspect").strip()
        with qualifier_col3:
            qualified_object_direction = st.text_input("Direction", 
                                                      value="", 
                                                      help="Direction of change (e.g., 'increased', 'decreased', 'upregulated')",
                                                      key="new_qualified_object_direction").strip()
        
        # Qualifier validation message for new interface
        has_any_qualifier = any([qualified_predicate.strip(), qualified_object_aspect.strip(), qualified_object_direction.strip()])
        if has_any_qualifier:
            if not qualified_predicate.strip():
                st.error("⚠️ Qualified Predicate is required when using any qualifiers")
            elif not qualified_object_aspect.strip() and not qualified_object_direction.strip():
                st.error("⚠️ At least one of Object Aspect or Direction must be provided when using qualifiers")
            else:
                # Build and show the qualified triple
                qualified_parts = []
                if qualified_object_direction.strip():
                    qualified_parts.append(qualified_object_direction.strip())
                if qualified_object_aspect.strip():
                    qualified_parts.append(qualified_object_aspect.strip())
                
                qualified_description = " ".join(qualified_parts)
                qualified_triple_str = f"'{subject}' {qualified_predicate.strip()} {qualified_description} of '{object_}'"
                st.success(f"✅ Qualified Triple: **{qualified_triple_str}**")
    
    # Initialize triple data structure
    triple_data = None
    
    if subject and predicate and object_:
        st.success(f"Triple: **{subject}** {predicate} **{object_}**")
        
        # Perform entity normalization if needed
        if input_mode == "Entity Names (with normalization)":
            with st.spinner("🧬 Normalizing entities..."):
                normalization_client = NodeNormalizationClient()
                
                # Get equivalent names
                subject_names = normalization_client.get_equivalent_names(name=subject)
                object_names = normalization_client.get_equivalent_names(name=object_)
                
                if not subject_names:
                    st.warning(f"⚠️ No equivalent names found for subject: {subject}")
                    subject_names = [subject]
                if not object_names:
                    st.warning(f"⚠️ No equivalent names found for object: {object_}")
                    object_names = [object_]
                
                # Show equivalent names found (always show the section for transparency)
                with st.expander("🔍 Equivalent Names Found"):
                    st.write(f"**{subject}** equivalent names:")
                    if len(subject_names) > 1:
                        for name in subject_names[:5]:  # Show first 5
                            st.write(f"• {name}")
                        if len(subject_names) > 5:
                            st.write(f"... and {len(subject_names) - 5} more")
                    else:
                        st.write(f"• {subject_names[0]} (primary name)")
                    
                    st.write(f"**{object_}** equivalent names:")
                    if len(object_names) > 1:
                        for name in object_names[:5]:  # Show first 5
                            st.write(f"• {name}")
                        if len(object_names) > 5:
                            st.write(f"... and {len(object_names) - 5} more")
                    else:
                        st.write(f"• {object_names[0]} (primary name)")
                
                triple_data = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_,
                    'subject_names': subject_names,
                    'object_names': object_names,
                    'qualified_predicate': qualified_predicate.strip() if qualified_predicate.strip() else None,
                    'qualified_object_aspect': qualified_object_aspect.strip() if qualified_object_aspect.strip() else None,
                    'qualified_object_direction': qualified_object_direction.strip() if qualified_object_direction.strip() else None
                }
        else:
            # CURIE mode - get equivalent names from CURIEs
            with st.spinner("🧬 Normalizing CURIEs..."):
                normalization_client = NodeNormalizationClient()
                
                # Get equivalent names from CURIEs
                subject_names = normalization_client.get_equivalent_names(curie=subject)
                object_names = normalization_client.get_equivalent_names(curie=object_)
                
                if not subject_names:
                    st.warning(f"⚠️ No equivalent names found for subject CURIE: {subject}")
                    subject_names = [subject]
                if not object_names:
                    st.warning(f"⚠️ No equivalent names found for object CURIE: {object_}")
                    object_names = [object_]
                
                # Show primary names and equivalents
                primary_subject = subject_names[0] if subject_names else subject
                primary_object = object_names[0] if object_names else object_
                
                st.info(f"🏷️ Resolved to: **{primary_subject}** {predicate} **{primary_object}**")
                
                # Always show equivalent names section for transparency
                with st.expander("🔍 Equivalent Names Found"):
                    st.write(f"**{subject}** equivalent names:")
                    if len(subject_names) > 1:
                        for name in subject_names[:5]:
                            st.write(f"• {name}")
                        if len(subject_names) > 5:
                            st.write(f"... and {len(subject_names) - 5} more")
                    else:
                        st.write(f"• {subject_names[0]} (primary name)")
                    
                    st.write(f"**{object_}** equivalent names:")
                    if len(object_names) > 1:
                        for name in object_names[:5]:
                            st.write(f"• {name}")
                        if len(object_names) > 5:
                            st.write(f"... and {len(object_names) - 5} more")
                    else:
                        st.write(f"• {object_names[0]} (primary name)")
                
                triple_data = {
                    'subject': primary_subject,
                    'predicate': predicate,
                    'object': primary_object,
                    'subject_names': subject_names,
                    'object_names': object_names,
                    'qualified_predicate': qualified_predicate.strip() if qualified_predicate.strip() else None,
                    'qualified_object_aspect': qualified_object_aspect.strip() if qualified_object_aspect.strip() else None,
                    'qualified_object_direction': qualified_object_direction.strip() if qualified_object_direction.strip() else None
                }
    
    st.divider()
    
    # PMID input section
    st.header("📄 PubMed IDs")
    
    input_tab1, input_tab2 = st.tabs(["📝 Manual Entry", "📁 File Upload"])
    
    pmids = []
    
    with input_tab1:
        pmid_input = st.text_area(
            "Enter PMIDs",
            value="34513929\n16488997\n14695375\n23613228\n34561318\n28473774\n26175950",
            height=150,
            help="Enter PubMed IDs separated by spaces, commas, or newlines"
        )
        
        if pmid_input.strip():
            # Parse PMIDs - handle spaces, commas, and newlines
            pmid_text = pmid_input.replace(',', ' ').replace('\n', ' ')
            pmids = [pmid.strip() for pmid in pmid_text.split() if pmid.strip()]
    
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload PMIDs file",
            type=['txt'],
            help="Upload a text file with one PMID per line"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            pmids = [line.strip() for line in content.split('\n') if line.strip()]
            st.success(f"Loaded {len(pmids)} PMIDs from file")
    
    # Display PMID preview
    if pmids:
        with st.expander(f"📊 Preview PMIDs ({len(pmids)} total)"):
            # Show first 20 PMIDs
            display_pmids = pmids[:20]
            cols = st.columns(5)
            for i, pmid in enumerate(display_pmids):
                with cols[i % 5]:
                    st.code(pmid)
            
            if len(pmids) > 20:
                st.info(f"... and {len(pmids) - 20} more PMIDs")
    
    st.divider()
    
    # Model selection
    st.header("🤖 AI Model Selection")
    
    # Get available models grouped by type
    ollama_models = settings.available_ollama_models
    openai_models = settings.available_openai_models
    available_models = settings.available_models

    if not available_models:
        st.error("No models configured. Set AVAILABLE_OLLAMA_MODELS and/or AVAILABLE_OPENAI_MODELS in .env or check your configuration.")
        st.stop()

    # Group models by type for display
    model_groups = []
    if ollama_models:
        model_groups.append(("🖥️ Ollama Models (Local)", ollama_models))
    if openai_models:
        model_groups.append(("☁️ OpenAI Models (Cloud)", openai_models))
    
    default_validation_model = settings.default_model
    # For checker, default to a different model if possible, otherwise use the same
    default_checker_model = available_models[1] if len(available_models) > 1 else available_models[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Validation Model")
        
        # Create grouped options with separators
        validation_options = []
        for group_name, group_models in model_groups:
            validation_options.append(f"--- {group_name} ---")
            validation_options.extend(group_models)
        
        # Find the index of the default model in the flattened list
        validation_index = 0
        for i, opt in enumerate(validation_options):
            if opt == default_validation_model:
                validation_index = i
                break
        
        model = st.selectbox(
            "Select validation model",
            options=validation_options,
            index=validation_index,
            format_func=lambda x: get_model_label(x) if not x.startswith("---") else x,
            help="Choose the language model for triple validation",
            key="validation_model_select",
            label_visibility="collapsed"
        )
        
        # Skip if user selected a separator
        while model.startswith("---"):
            st.warning("Please select an actual model, not a separator")
            model = available_models[0]
        
        show_model_info(model)
    
    with col2:
        ver_col1, ver_col2 = st.columns([3, 1])
        with ver_col1:
            st.subheader("Verification Model")
        with ver_col2:
            enable_verification = st.checkbox(
                "Enable",
                value=False,
                help="Use a second model to double-check name recognition"
            )
        
        checker_model = None
        if enable_verification:
            # Create grouped options for checker
            checker_options = []
            for group_name, group_models in model_groups:
                checker_options.append(f"--- {group_name} ---")
                checker_options.extend(group_models)
            
            # Find the index of the default checker model
            checker_index = 0
            for i, opt in enumerate(checker_options):
                if opt == default_checker_model:
                    checker_index = i
                    break
            
            checker_model = st.selectbox(
                "Select checker model",
                options=checker_options,
                index=checker_index,
                format_func=lambda x: get_model_label(x) if not x.startswith("---") else x,
                help="Model used to verify validation results",
                key="checker_model_select",
                label_visibility="collapsed"
            )
            
            # Skip if user selected a separator
            while checker_model.startswith("---"):
                st.warning("Please select an actual model, not a separator")
                checker_model = available_models[0]
            
            show_model_info(checker_model)
        else:
            st.warning("⚠️ **No Verification Enabled**\n\n"
                      "Without verification:\n\n"
                      "• Entity names checked once only\n\n"
                      "• May miss some synonyms\n\n"
                      "• Results returned faster")
    
    
    st.divider()
    
    # Evaluation section
    qualifiers_valid = True
    if has_any_qualifier:
        qualifiers_valid = (qualified_predicate.strip() and 
                          (qualified_object_aspect.strip() or qualified_object_direction.strip()))
    
    can_evaluate = triple_data is not None and pmids and qualifiers_valid
    
    if can_evaluate:
        if st.button("🚀 Start Check", type="primary"):
            
            # Check environment first
            env_status = check_environment()
            if env_status["missing_vars"]:
                st.error(f"❌ Missing required environment variables: {', '.join(env_status['missing_vars'])}")
                with st.expander("📋 Setup Instructions"):
                    st.markdown("""
                    **Create a .env file with:**
                    ```
                    NCBI_EMAIL=your.email@example.com
                    NCBI_API_KEY=your_api_key_here
                    OLLAMA_BASE_URL=http://localhost:11434
                    ```
                    """)
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(progress_value, message):
                progress_bar.progress(progress_value)
                status_text.text(message)
            
            try:
                # Run the evaluation with progress tracking
                result = asyncio.run(run_evaluation(triple_data, pmids, model, checker_model, update_progress))
                
                # Store in session state
                st.session_state.last_result = result
                st.session_state.last_triple = [triple_data['subject'], triple_data['predicate'], triple_data['object']]
                st.session_state.last_model = model
                
                # Clear progress indicators after a short delay
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Rerun to show results
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Evaluation failed: {str(e)}")
                
                # Detailed error information
                with st.expander("🐛 Error Details"):
                    st.code(str(e))
                    st.markdown("**Troubleshooting:**")
                    st.markdown("- Make sure Ollama is running: `./setup_ollama_pmid_support.sh`")
                    st.markdown("- Check if the model is available: `ollama list`")
                    st.markdown("- Verify your .env configuration")
    
    else:
        st.info("👆 Please fill in all fields to start check")
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        st.divider()
        display_evaluation_results()

def display_evaluation_results():
    """Display the evaluation results with charts and detailed breakdown."""
    result = st.session_state.last_result
    triple = st.session_state.last_triple
    model = st.session_state.last_model
    
    st.header("📈 Results")
    
    # Summary
    summary = result.get_summary()
    st.subheader(f"Triple: **{triple[0]}** {triple[1]} **{triple[2]}** (Model: {model})")
    
    # Layout: 2x2 metrics on left, pie chart on right
    metrics_col, chart_col = st.columns([1, 1])
    
    with metrics_col:
        # Top row of metrics
        metric_row1_col1, metric_row1_col2 = st.columns(2)
        with metric_row1_col1:
            st.metric("Total PMIDs", summary['total_pmids'])
        with metric_row1_col2:
            st.metric("✅ Supported", summary['supported_pmids'], delta=f"{summary['supported_percentage']}%")
        
        # Bottom row of metrics
        metric_row2_col1, metric_row2_col2 = st.columns(2)
        with metric_row2_col1:
            st.metric("❌ Not Supported", summary['unsupported_pmids'], delta=f"{summary['unsupported_percentage']}%")
        with metric_row2_col2:
            # Count direct support PMIDs
            direct_support_count = len([e for e in result.evaluations if e.evidence_category == "direct_support"])
            st.metric("🎯 Direct Support", direct_support_count)
    
    with chart_col:
        # Create and display pie chart
        df = create_results_dataframe(result)
        if not df.empty:
            create_summary_charts(df)
    
    # Results table with tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 All Results", "✅ Supported Only", "❌ Not Supported Only"])
    
    with tab1:
        display_results_table(df, "all")
    
    with tab2:
        supported_df = df[df['Supported']]
        if not supported_df.empty:
            display_results_table(supported_df, "supported")
        else:
            st.info("No supported PMIDs found.")
    
    with tab3:
        not_supported_df = df[~df['Supported']]
        if not not_supported_df.empty:
            display_results_table(not_supported_df, "not_supported")
        else:
            st.info("All PMIDs are supported!")
    
    # Export section
    st.subheader("📥 Export Results")
    export_results(result, triple, model)

def display_results_table(df: pd.DataFrame, view_type: str):
    """Display results table with appropriate styling."""
    if df.empty:
        st.info("No results to display.")
        return
    
    # Configure columns with tooltips
    column_config = {
        'PMID': st.column_config.LinkColumn(
            'PMID', 
            help="PubMed ID - Click to view the full article on PubMed",
            display_text=r"https://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)"
        ),
        'Supported': st.column_config.CheckboxColumn(
            'Supported', 
            help="Whether the triple relationship is supported (Direct support and Implicit support count as supported)"
        ),
        'Category': st.column_config.TextColumn(
            'Category', 
            help="Evidence category:\n• Direct support = exact/semantic match\n• Implicit support = general predicate match\n• Opposite Assertion = opposite predicate\n• Wrong qualifier = conflicts with qualifiers\n• Missing qualifier = missing qualifiers\n• Not supported = not mentioned\n• Unknown = ambiguous"
        ),
        'Subject Mentioned': st.column_config.TextColumn(
            'Subject Mentioned', 
            help="Whether the subject entity (or its equivalent names) is mentioned in the abstract"
        ),
        'Object Mentioned': st.column_config.TextColumn(
            'Object Mentioned', 
            help="Whether the object entity (or its equivalent names) is mentioned in the abstract"
        ),
        'Supporting Sentence': st.column_config.TextColumn(
            'Supporting Sentence', 
            help="The most relevant sentence from the abstract (shown for Direct support and Implicit support)", 
            width="large"
        ),
        'Reasoning': st.column_config.TextColumn(
            'Reasoning', 
            help="Detailed explanation of the categorization decision", 
            width="large"
        )
    }
    
    # Display with styling
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        height=min(400, len(df) * 35 + 50)
    )

def export_results(result: TripleEvaluationResult, triple: List[str], model: str):
    """Provide export options for results."""
    col1, col2, col3 = st.columns(3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        # CSV export
        df = create_results_dataframe(result)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"pmid_evaluation_{timestamp}.csv",
            mime="text/csv",
        )
    
    with col2:
        # JSON export
        summary = result.get_summary()
        json_data = {
            "metadata": {
                "triple": {"subject": triple[0], "predicate": triple[1], "object": triple[2]},
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "total_pmids": summary['total_pmids']
            },
            "summary": summary,
            "evaluations": [
                {
                    "pmid": e.pmid,
                    "is_supported": e.is_supported,
                    "evidence_category": e.evidence_category,
                    "supporting_sentence": e.supporting_sentence,
                    "reasoning": e.reasoning,
                    "subject_mentioned": e.subject_mentioned,
                    "object_mentioned": e.object_mentioned
                }
                for e in result.evaluations
            ]
        }
        
        st.download_button(
            label="📋 Download as JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"pmid_evaluation_{timestamp}.json",
            mime="application/json",
        )
    
    with col3:
        # CLI format export
        cli_output = result.format_output()
        st.download_button(
            label="🖥️ CLI Format",
            data=cli_output,
            file_name=f"pmid_evaluation_{timestamp}.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    run_app()