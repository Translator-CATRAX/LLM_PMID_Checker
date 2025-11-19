#!/usr/bin/env python3
"""Main interface for triple checking."""
import argparse
import asyncio
import logging
import sys
from src.triple_evaluator import TripleEvaluatorSystem
from src.node_normalization import NodeNormalizationClient
from src.config import settings

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Check research triples against PMID abstracts using Ollama or OpenAI LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=        """
Examples:
  # Basic triples using names (requires node normalization)
  # Using Ollama models (local)
  python main.py --val_model gpt-oss:20b --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  python main.py --val_model hermes4:70b-q4-m --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  
  # Using OpenAI models (cloud)
  python main.py --val_model gpt-5-nano --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  python main.py --val_model gpt-5-mini --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  
  # Basic triples using CURIEs directly
  python main.py --val_model gpt-oss:20b --triple_curie "NCBIGene:6495" "affects" "UMLS:C0596290" --pmids 16186693 29083299
  python main.py --val_model gpt-5-nano --triple_curie "NCBIGene:6495" "affects" "UMLS:C0596290" --pmids 16186693 29083299
  
  # With qualifiers - must provide qualified_predicate and at least one of qualified_object_aspect/qualified_object_direction
  python main.py --val_model hermes4:70b-q4-m --triple_name "SIX1" "affects" "Cell Proliferation" --qualified_predicate "causes" --qualified_object_aspect "activity" --qualified_object_direction "increased" --pmids 16186693 29083299
  python main.py --val_model gpt-5-mini --triple_curie "NCBIGene:6495" "affects" "UMLS:C0596290" --qualified_predicate "causes" --qualified_object_direction "upregulated" --pmids 16186693 29083299
  
  # With verification (mixing local and cloud models)
  python main.py --val_model hermes4:70b-q4-m --checker_model gpt-oss:20b --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  python main.py --val_model gpt-5-nano --checker_model gpt-5-mini --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
        """
    )
    
    # Triple specification - mutually exclusive options
    triple_group = parser.add_mutually_exclusive_group(required=True)
    triple_group.add_argument(
        '--triple_curie', 
        nargs=3,
        metavar=('SUBJECT_CURIE', 'PREDICATE', 'OBJECT_CURIE'),
        help='Research triple as CURIEs (e.g., "NCBIGene:6495" "affects" "UMLS:C0596290")'
    )
    triple_group.add_argument(
        '--triple_name', 
        nargs=3,
        metavar=('SUBJECT_NAME', 'PREDICATE', 'OBJECT_NAME'),
        help='Research triple as names (e.g., "SIX1" "affects" "Cell Proliferation")'
    )
    
    # Qualifier options
    parser.add_argument(
        '--qualified_predicate',
        help='Qualified predicate (e.g., "causes"). Required if any qualifier is used.'
    )
    parser.add_argument(
        '--qualified_object_aspect',
        help='Object aspect qualifier (e.g., "activity", "abundance", "activity_or_abundance"). Optional.'
    )
    parser.add_argument(
        '--qualified_object_direction',
        help='Object direction qualifier (e.g., "increased", "decreased", "upregulated", "downregulated"). Optional.'
    )
    
    # PMID specification
    pmid_group = parser.add_mutually_exclusive_group(required=True)
    pmid_group.add_argument(
        '--pmids',
        nargs='+',
        help='List of PMIDs to evaluate'
    )
    pmid_group.add_argument(
        '--pmids-file',
        help='File containing PMIDs (one per line)'
    )
    
    # Model selection (required)
    parser.add_argument('--val_model',
                               type=str,
                               default=settings.default_model,
                               help=f"Model for triple validation (available: {', '.join(settings.available_models)}).")
    
    parser.add_argument('--checker_model',
                               type=str,
                               default=None,
                               help=f"Model for verification/checking equivalent names. If not provided, verification is disabled. Available: {', '.join(settings.available_models)}")
    
    # Optional arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate qualifier constraints
    has_any_qualifier = any([
        args.qualified_predicate,
        args.qualified_object_aspect,
        args.qualified_object_direction
    ])
    
    if has_any_qualifier:
        if not args.qualified_predicate:
            print("Error: qualified_predicate is required when using any qualifiers", file=sys.stderr)
            return 1
        
        if not args.qualified_object_aspect and not args.qualified_object_direction:
            print("Error: At least one of qualified_object_aspect or qualified_object_direction must be provided when using qualifiers", file=sys.stderr)
            return 1
    
    # Parse triple and get equivalent names
    normalization_client = NodeNormalizationClient()
    
    if args.triple_curie:
        subject_curie, predicate, object_curie = args.triple_curie
        print(f"Getting equivalent names for CURIEs: {subject_curie}, {object_curie}")
        
        # Get equivalent names for subject and object
        subject_names = normalization_client.get_equivalent_names(curie=subject_curie)
        object_names = normalization_client.get_equivalent_names(curie=object_curie)
        
        if not subject_names:
            print(f"Error: No equivalent names found for subject CURIE: {subject_curie}", file=sys.stderr)
            subject_names = [subject_curie]
        if not object_names:
            print(f"Warning: No equivalent names found for object CURIE: {object_curie}", file=sys.stderr)
            object_names = [object_curie]
            
        triple = [subject_names[0], predicate, object_names[0]]  # Use primary name for display
        triple_with_names = {
            'subject': subject_names[0],
            'predicate': predicate,
            'object': object_names[0],
            'subject_names': subject_names,
            'object_names': object_names
        }
        
    elif args.triple_name:
        subject_name, predicate, object_name = args.triple_name
        subject_name = subject_name.replace(',','')
        object_name = object_name.replace(',','')
        print(f"Getting equivalent names for names: {subject_name}, {object_name}")
        
        # Get equivalent names for subject and object
        subject_names = normalization_client.get_equivalent_names(name=subject_name)
        object_names = normalization_client.get_equivalent_names(name=object_name)
        
        if not subject_names:
            print(f"Warning: No equivalent names found for subject name: {subject_name}", file=sys.stderr)
            subject_names = [subject_name]
        if not object_names:
            print(f"Warning: No equivalent names found for object name: {object_name}", file=sys.stderr)
            object_names = [object_name]
            
        triple = [subject_name, predicate, object_name]  # Use original name for display
        triple_with_names = {
            'subject': subject_name,
            'predicate': predicate,
            'object': object_name,
            'subject_names': subject_names,
            'object_names': object_names
        }
    else:
        print("Error: Either --triple_curie or --triple_name must be provided", file=sys.stderr)
        return 1
    
    # Parse PMIDs
    if args.pmids:
        pmids = args.pmids
    else:
        try:
            with open(args.pmids_file, 'r') as f:
                pmids = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"PMIDs file not found: {args.pmids_file}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error reading PMIDs file: {e}", file=sys.stderr)
            return 1
    
    if not pmids:
        print("No PMIDs provided", file=sys.stderr)
        return 1
    
    print(f"Checking triple {triple} against {len(pmids)} PMIDs...")
    print("=" * 60)
    
    try:
        # Validate models
        try:
            validation_model = settings.validate_model(args.val_model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        checker_model = None
        if args.checker_model:
            try:
                checker_model = settings.validate_model(args.checker_model)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        
        # Log model configuration
        logger.info(f"Using validation model: {validation_model}")
        if checker_model:
            logger.info(f"Using checker model: {checker_model}")
            logger.info(f"Verification enabled: True")
        else:
            logger.info(f"Verification disabled (no checker model provided)")
        
        print(f"Validation model: {validation_model}")
        if checker_model:
            print(f"Checker model: {checker_model}")
        else:
            print("Verification disabled")
        print("=" * 60)
        
        # Create evaluator system with specified model
        evaluator = TripleEvaluatorSystem(
            llm_provider=validation_model,
            checker_model=checker_model
        )
        
        # Run the evaluation with enriched triple data
        results = await evaluator.evaluate_triple_with_names(
            subject=triple_with_names['subject'],
            predicate=triple_with_names['predicate'], 
            object_=triple_with_names['object'],
            subject_names=triple_with_names['subject_names'],
            object_names=triple_with_names['object_names'],
            pmids=pmids,
            qualified_predicate=args.qualified_predicate,
            qualified_object_aspect=args.qualified_object_aspect,
            qualified_object_direction=args.qualified_object_direction
        )
        
        # Output results
        formatted_output = results.format_output(verbose=args.verbose) if hasattr(results, 'format_output') else str(results)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            print(f"Results written to {args.output}")
        else:
            print(formatted_output)
        
        # Print final summary
        summary = results.get_summary()
        print("\n" + "=" * 60)
        print("CHECK SUMMARY")
        print("=" * 60)
        print(f"Total PMIDs: {summary['total_pmids']}")
        print(f"Supported: {summary['supported_pmids']} ({summary['supported_percentage']}%)")
        print(f"Not Supported: {summary['unsupported_pmids']} ({summary['unsupported_percentage']}%)")
        
        return 0
        
    except Exception as e:
        print(f"Error during check: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))