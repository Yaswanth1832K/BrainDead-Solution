"""
Text quality evaluation (simplified BLEU)
"""

from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)
