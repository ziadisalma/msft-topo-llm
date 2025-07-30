import numpy as np
import torch
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datetime import datetime

def set_up_dir(path):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def create_3sat_cot_prompt(n_vars, n_clauses, clauses):
    """Create Chain of Thought prompt for 3-SAT problem"""
    
    # Format clauses for display
    clause_strings = []
    for i, clause in enumerate(clauses):
        literals = []
        for literal in clause:
            if literal > 0:
                literals.append(f"x_{literal}")
            else:
                literals.append(f"¬x_{abs(literal)}")
        clause_strings.append(f"C{i+1} = ({' ∨ '.join(literals)})")
    
    clauses_display = '\n     '.join(clause_strings)
    
    prompt = f"""You are given the following Boolean formula in 3-CNF form:

n_vars: {n_vars}
n_clauses: {n_clauses}
clauses:
{clauses}

Each clause [a, b, c] represents (literal a ∨ literal b ∨ literal c), where:
- a positive integer k means variable x_k
- a negative integer –k means ¬x_k

Your task is to determine whether this formula is satisfiable, and if so, to produce one explicit satisfying assignment.

**Show your full chain-of-thought**—explain each decision, unit propagation, and backtracking step.

---

Begin like this:
1. **List variables and clauses.**
   - Variables: x₁…x_{n_vars}
   - Clauses:
     {clauses_display}

2. **Decision 1:** Choose a value for x₁.
3. **Propagate:** Simplify or satisfy clauses based on that choice.
4. **Decision 2:** Choose a value for another variable.
5. **Propagate again:** Simplify remaining clauses.
6. **Check for conflicts:** If conflict, backtrack and flip the last decision.
7. **Repeat** until all clauses are satisfied or you prove unsatisfiability.
8. **Report:**
   - State "SATISFIABLE" or "UNSATISFIABLE"
   - If satisfiable, give a full assignment {{x₁=..., x₂=..., x₃=..., etc.}}. Remember you can only assign 0 or 1 to each variable with 0 meaning false and 1 meaning true. No other values are allowed

Now solve it, showing every step:
"""
    return prompt

def extract_3sat_answer(text):
    """Extract satisfiability result and assignment from model output"""
    if not text:
        return None, None
    
    text = text.strip().upper()
    
    # Check for satisfiability
    is_satisfiable = None
    if "SATISFIABLE" in text and "UNSATISFIABLE" not in text:
        is_satisfiable = True
    elif "UNSATISFIABLE" in text:
        is_satisfiable = False
    elif "SAT" in text and "UNSAT" not in text:
        is_satisfiable = True
    elif "UNSAT" in text:
        is_satisfiable = False
    
    # Extract assignment if satisfiable
    assignment = None
    if is_satisfiable:
        # Look for assignment patterns like {x1=True, x2=False, ...} or x1=1, x2=0, etc.
        assignment_patterns = [
            r'x[_]?(\d+)\s*=\s*(True|1|T)',
            r'x[_]?(\d+)\s*=\s*(False|0|F)',
            r'X[_]?(\d+)\s*=\s*(True|1|T)',
            r'X[_]?(\d+)\s*=\s*(False|0|F)'
        ]
        
        assignment = {}
        for pattern in assignment_patterns:
            matches = re.findall(pattern, text)
            for var_num, value in matches:
                var_num = int(var_num)
                if value in ['True', '1', 'T']:
                    assignment[var_num] = True
                else:
                    assignment[var_num] = False
    
    return is_satisfiable, assignment

def verify_assignment(clauses, assignment, n_vars):
    """Verify if the given assignment satisfies all clauses"""
    if not assignment or len(assignment) != n_vars:
        return False
    
    for clause in clauses:
        clause_satisfied = False
        for literal in clause:
            var_num = abs(literal)
            if var_num in assignment:
                if literal > 0 and assignment[var_num]:  # Positive literal and variable is True
                    clause_satisfied = True
                    break
                elif literal < 0 and not assignment[var_num]:  # Negative literal and variable is False
                    clause_satisfied = True
                    break
        
        if not clause_satisfied:
            return False
    
    return True

def run_3sat_benchmark(models, num_samples=50, output_dir="results_3sat"):
    """Run 3SAT benchmark on specified models"""
    set_up_dir(output_dir)
    
    # Load dataset
    print("Loading 3SAT dataset...")
    # Load full dataset first
    full_dataset = load_dataset("khoomeik/satscale-3-sat-20000", split="train")

    # Filter for problems with n_vars < 13
    filtered_dataset = full_dataset.filter(lambda x: x['n_vars'] < 13)

    # Select the first num_samples from filtered dataset
    dataset = filtered_dataset.select(range(min(num_samples, len(filtered_dataset))))

    print(f"Total problems with n_vars < 13: {len(filtered_dataset)}")
    print(f"Selected {len(dataset)} problems for testing")
    
    all_results = []
    
    for model_name, config in models.items():
        print(f"\nRunning benchmark for {model_name}...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        
        model_results = []
        
        for idx, item in enumerate(dataset):
            name = item['name']
            n_vars = item['n_vars']
            n_clauses = item['n_clauses']
            clauses = item['clauses']
            true_assignment = item['assignments']
            marginals = item['marginals']
            
            # Create prompt
            prompt = create_3sat_cot_prompt(n_vars, n_clauses, clauses)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[-1]
            
            gen_kwargs = {
                'max_new_tokens': 2000,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'pad_token_id': tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            answer_text = full_text[len(prompt):].strip()
            
            # Extract answer
            is_satisfiable, predicted_assignment = extract_3sat_answer(answer_text)
            
            # Calculate metrics
            gen_len = gen_ids.shape[-1] - input_len
            
            # Verify if the predicted assignment is correct
            assignment_correct = False
            if is_satisfiable and predicted_assignment:
                assignment_correct = verify_assignment(clauses, predicted_assignment, n_vars)
            
            # Convert true assignment to dictionary format for comparison
            true_assignment_dict = {i+1: bool(val) for i, val in enumerate(true_assignment)}
            
            # Record result
            result = {
                'model': model_name,
                'idx': idx,
                'name': name,
                'n_vars': n_vars,
                'n_clauses': n_clauses,
                'clauses': clauses,
                'true_assignment': true_assignment_dict,
                'marginals': marginals,
                'prompt': prompt,
                'full_response': answer_text,
                'predicted_satisfiable': is_satisfiable,
                'predicted_assignment': predicted_assignment,
                'assignment_correct': assignment_correct,
                'tokens_generated': gen_len,
                'timestamp': datetime.now().isoformat()
            }
            
            model_results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{num_samples}")
        
        # Save individual model results
        model_output_file = os.path.join(output_dir, f"{model_name}_results.json")
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        all_results.extend(model_results)
        
        # Print summary for this model
        correct_count = sum(1 for r in model_results if r['assignment_correct'])
        satisfiable_predictions = sum(1 for r in model_results if r['predicted_satisfiable'])
        total_problems = len(model_results)
        
        print(f"  Results for {model_name}:")
        print(f"    Correct assignments: {correct_count}/{total_problems} ({correct_count/total_problems:.3f})")
        print(f"    Predicted satisfiable: {satisfiable_predictions}/{total_problems} ({satisfiable_predictions/total_problems:.3f})")
        print(f"    Average tokens generated: {np.mean([r['tokens_generated'] for r in model_results]):.1f}")
    
    # Save combined results
    combined_output_file = os.path.join(output_dir, "all_results.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results

def analyze_results(all_results, output_dir):
    """Analyze and compare model performance"""
    
    # Group results by model
    model_results = {}
    for result in all_results:
        model_name = result['model']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    # Calculate summary statistics
    summary = {}
    for model_name, results in model_results.items():
        total = len(results)
        correct = sum(1 for r in results if r['assignment_correct'])
        satisfiable_pred = sum(1 for r in results if r['predicted_satisfiable'])
        avg_tokens = np.mean([r['tokens_generated'] for r in results])
        
        summary[model_name] = {
            'total_problems': total,
            'correct_assignments': correct,
            'accuracy': correct / total,
            'satisfiable_predictions': satisfiable_pred,
            'satisfiable_rate': satisfiable_pred / total,
            'avg_tokens_generated': avg_tokens
        }
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print comparison
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    
    for model_name, stats in summary.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {stats['accuracy']:.3f} ({stats['correct_assignments']}/{stats['total_problems']})")
        print(f"  Satisfiable Rate: {stats['satisfiable_rate']:.3f}")
        print(f"  Avg Tokens: {stats['avg_tokens_generated']:.1f}")
    
    return summary

def main():
    """Main execution function"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"results_3sat_{timestamp}"
    
    # Model configurations
    models = {
        'phi4_cot': {
            'path': 'microsoft/phi-4',
            'mode': 'cot'
        },
        'phi4r_cot': {
            'path': 'microsoft/Phi-4-reasoning',
            'mode': 'cot'
        }
    }
    
    print("Starting 3-SAT benchmark comparison...")
    print(f"Results will be saved to: {base_dir}")
    
    # Run benchmark
    all_results = run_3sat_benchmark(
        models=models,
        num_samples=50,
        output_dir=base_dir
    )
    
    # Analyze results
    summary = analyze_results(all_results, base_dir)
    
    print(f"\nAll results saved to: {base_dir}")
    print("Files created:")
    print(f"  - phi4_cot_results.json")
    print(f"  - phi4r_cot_results.json")
    print(f"  - all_results.json")
    print(f"  - summary.json")

if __name__ == '__main__':
    main()