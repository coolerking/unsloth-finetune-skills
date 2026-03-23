"""Unsloth Fine-Tuning Orchestrator - Orchestrate the complete fine-tuning workflow.

This module provides functionality to:
- Create run directories with timestamps
- Orchestrate dataset creation via unsloth_dataset_creator
- Orchestrate model fine-tuning via unsloth_trainer
- Handle errors at each step
- Return comprehensive workflow results
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent skills to Python path
SKILLS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILLS_DIR))

# Import child skills
from unsloth_dataset_creator import create_dataset
from unsloth_trainer import fine_tune

__all__ = [
    'run_workflow',
    'create_run_directory',
]


def create_run_directory(output_dir: str) -> str:
    """Create a run directory with timestamp.

    Args:
        output_dir: Base output directory path

    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def run_workflow(
    pdf_dir: str,
    output_dir: str,
    base_model: str,
    config: Dict[str, Any],
    llm_provider: str,
    llm_config: Dict[str, Any],
    optuna_config: Optional[Dict[str, Any]] = None,
    auto_improve_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the complete fine-tuning workflow.

    This function orchestrates the entire workflow from PDF documents to
    a fine-tuned model. It performs dataset creation, model training, and
    evaluation in sequence.

    Args:
        pdf_dir: Directory containing PDF files to process
        output_dir: Base directory for all outputs
        base_model: Name or path of the base model to fine-tune
        config: Configuration for dataset creation (use_thinking, target_samples, etc.)
        llm_provider: LLM provider name (e.g., "groq")
        llm_config: LLM configuration dict (api_key, model, temperature, etc.)
        optuna_config: Optional configuration for Optuna hyperparameter optimization.
            If None, uses default config with n_trials=20.
        auto_improve_config: Optional configuration for auto-improvement loop.
            Currently a placeholder for future implementation.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - run_id: Unique identifier for this workflow run
        - paths: Dictionary of important file paths
        - metadata: Workflow metadata including dataset and training info
        - error: Error message (if status is "error")

    Example:
        >>> result = run_workflow(
        ...     pdf_dir="/path/to/pdfs",
        ...     output_dir="/path/to/output",
        ...     base_model="unsloth/Llama-3.2-3B-Instruct",
        ...     config={"use_thinking": True, "target_samples": 100},
        ...     llm_provider="groq",
        ...     llm_config={"api_key": "xxx", "model": "llama-3.1-8b-instant"}
        ... )
        >>> print(result["status"])
        "success"
    """
    # Generate run_id and create run directory
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Starting Fine-Tuning Workflow")
    print(f"Run ID: {run_id}")
    print(f"Output Directory: {run_dir}")
    print(f"=" * 60)

    # Initialize result structure
    result = {
        'status': 'success',
        'run_id': run_id,
        'paths': {
            'run_dir': str(run_dir),
            'dataset_dir': str(run_dir / 'dataset'),
            'training_dir': str(run_dir / 'training'),
            'evaluation_dir': str(run_dir / 'evaluation'),
        },
        'metadata': {
            'base_model': base_model,
            'pdf_dir': pdf_dir,
            'config': config,
            'llm_provider': llm_provider,
        },
        'dataset_result': None,
        'training_result': None,
        'evaluation_result': None,
        'error': None
    }

    # =======================================================================
    # Step 1: Create Dataset
    # =======================================================================
    print("\n[Step 1/3] Creating Dataset...")
    dataset_dir = run_dir / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_result = create_dataset(
            pdf_dir=pdf_dir,
            output_dir=str(dataset_dir),
            config=config,
            llm_provider=llm_provider,
            llm_config=llm_config
        )

        if dataset_result.get('status') != 'success':
            error_msg = dataset_result.get('message', 'Unknown error in dataset creation')
            print(f"ERROR: Dataset creation failed: {error_msg}")
            result['status'] = 'error'
            result['error'] = f"Dataset creation failed: {error_msg}"
            result['dataset_result'] = dataset_result
            return result

        print(f"SUCCESS: Dataset created with {dataset_result.get('metadata', {}).get('total_samples', 0)} samples")
        result['dataset_result'] = dataset_result

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Dataset creation failed with exception: {error_msg}")
        result['status'] = 'error'
        result['error'] = f"Dataset creation failed: {error_msg}"
        return result

    # =======================================================================
    # Step 2: Fine-tune Model
    # =======================================================================
    print("\n[Step 2/3] Fine-tuning Model...")
    training_dir = run_dir / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = str(dataset_dir / 'dataset_train.jsonl')
    eval_dataset = str(dataset_dir / 'dataset_eval.jsonl')

    try:
        training_result = fine_tune(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(training_dir),
            base_model=base_model,
            optuna_config=optuna_config
        )

        if training_result.get('status') != 'success':
            error_msg = training_result.get('error', 'Unknown error in training')
            print(f"ERROR: Training failed: {error_msg}")
            result['status'] = 'error'
            result['error'] = f"Training failed: {error_msg}"
            result['training_result'] = training_result
            return result

        print(f"SUCCESS: Model trained successfully")
        print(f"  Best model path: {training_result.get('best_model_path')}")
        print(f"  Best eval loss: {training_result.get('best_eval_loss')}")
        result['training_result'] = training_result

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Training failed with exception: {error_msg}")
        result['status'] = 'error'
        result['error'] = f"Training failed: {error_msg}"
        return result

    # =======================================================================
    # Step 3: Evaluate (Placeholder)
    # =======================================================================
    print("\n[Step 3/3] Evaluation...")
    evaluation_dir = run_dir / 'evaluation'
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for evaluation step
    # TODO: Implement auto-improver integration in Task 5
    print("NOTE: Evaluation step is currently a placeholder.")
    print("      Auto-improver integration will be implemented in Task 5.")

    result['evaluation_result'] = {
        'status': 'placeholder',
        'message': 'Evaluation not yet implemented. Auto-improver coming in Task 5.'
    }

    # =======================================================================
    # Workflow Complete
    # =======================================================================
    print("\n" + "=" * 60)
    print(f"Workflow Complete - Status: {result['status']}")
    print(f"=" * 60)

    return result
