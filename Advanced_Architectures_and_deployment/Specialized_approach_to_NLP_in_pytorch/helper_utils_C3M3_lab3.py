import os
import urllib.request
from tqdm import tqdm
import torch

def get_shakespeare_data(filename="shakespeare.txt", data_dir="./"):
    """
    Downloads and loads the Shakespeare dataset.
    
    Parameters:
    -----------
    filename : str, optional
        Name for the saved file (default: 'shakespeare.txt')
    data_dir : str, optional
        Directory to save the file (default: current directory)
    
    Returns:
    --------
    str
        The complete Shakespeare text
    """
    # Create full file path
    filepath = os.path.join(data_dir, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"Shakespeare dataset already exists at {filepath}")
    else:
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Download the file
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Download complete! Saved to {filepath}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise
    
    # Read and return the text
    print("Loading Shakespeare text...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Text loaded successfully! ({len(text):,} characters)")
    print(f"Preview: {text[:300]}...")  # Show preview
    
    return text


def train_model(model, vocab_size, loader, loss_fn, optimizer, epochs=10, device='cpu'):
    """Train the decoder model on Shakespeare text"""
    model.to(device)  # Ensure model is on the right device
    
    for epoch in range(epochs):
        model.train()  # Set to training mode
        epoch_losses = []  # Track losses for averaging
        
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move batch to device
                xb, yb = xb.to(device), yb.to(device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass through decoder
                logits = model(xb)  # Shape: [batch, seq_len, vocab_size]
                
                # Reshape for loss calculation
                loss = loss_fn(
                    logits.reshape(-1, vocab_size),  # [batch*seq_len, vocab_size]
                    yb.reshape(-1)                    # [batch*seq_len]
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping (ADD THIS!)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Track loss
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average loss - simple mean
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1:2d}: avg loss = {avg_loss:.4f}")


# helper_utils.py

import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class ShakespeareTokenizer:
    """Tokenizer for Shakespeare text that handles contractions and punctuation"""
    def __call__(self, text):
        # Replace line breaks with a special token
        text = text.replace('\n', ' <nl> ')
        # Tokenize words, contractions, <nl>, and punctuation
        return re.findall(r"\w+(?:'\w+)?|<nl>|[^\w\s]", text)

def build_vocabulary(text, vocab_size=5000, tokenizer=None):
    """
    Build vocabulary from Shakespeare text using top-k most frequent tokens.
    
    Args:
        text: Raw Shakespeare text
        vocab_size: Maximum vocabulary size (default: 5000)
        tokenizer: Tokenizer instance (if None, creates ShakespeareTokenizer)
    
    Returns:
        vocab: List of vocabulary words
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
        tokenizer: The tokenizer used
    """
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Count all tokens
    tokens = tokenizer(text)
    token_counts = Counter(tokens)
    
    # Always include special tokens
    special_tokens = ['<pad>', '<unk>', '<nl>']
    
    # Get top-k most frequent tokens (excluding space for special tokens)
    most_common = token_counts.most_common(vocab_size - len(special_tokens))
    
    # Build vocab - special tokens first, then top frequent tokens
    vocab = special_tokens.copy()
    for token, count in most_common:
        if token not in special_tokens:
            vocab.append(token)
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Calculate coverage statistics
    total_token_occurrences = sum(token_counts.values())
    covered_token_occurrences = sum(token_counts[token] for token in vocab if token in token_counts)
    coverage = covered_token_occurrences / total_token_occurrences
    
    # Calculate unknown rate
    unknown_count = sum(count for token, count in token_counts.items() if token not in word2idx)
    unknown_rate = unknown_count / total_token_occurrences
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tokens in text: {len(token_counts)}")
    print(f"Coverage: {coverage:.1%} of token occurrences")
    print(f"Unknown token rate: {unknown_rate:.1%}")
    print(f"Most common tokens: {vocab[3:13]}")
    print(f"Least common in vocab: {vocab[-10:]}")
    
    return vocab, word2idx, idx2word, tokenizer

def create_sequences(text, word2idx, idx2word, tokenizer=None, seq_len=150):
    """
    Create training sequences from text.
    
    Args:
        text: Raw Shakespeare text
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        tokenizer: Tokenizer instance (if None, creates ShakespeareTokenizer)
        seq_len: Length of each sequence (default: 150)
    
    Returns:
        inputs: List of input sequences
        targets: List of target sequences (shifted by 1)
    """
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()
    
    # Tokenize the full text
    tokens = tokenizer(text)
    
    inputs = []
    targets = []
    
    # Create sliding windows
    for i in range(len(tokens) - seq_len):
        # Extract window and target (shifted by 1)
        window = tokens[i:i+seq_len]
        target = tokens[i+1:i+seq_len+1]
        
        # Convert to indices
        input_ids = [word2idx.get(w, word2idx['<unk>']) for w in window]
        target_ids = [word2idx.get(w, word2idx['<unk>']) for w in target]
        
        inputs.append(input_ids)
        targets.append(target_ids)
    
    print(f"Created {len(inputs)} sequences of length {seq_len}")
    
    # Show example with actual tokens for verification
    if inputs:
        # Show the actual tokens
        input_tokens = [idx2word[id] for id in inputs[0][:10]]
        target_tokens = [idx2word[id] for id in targets[0][:10]]
        
        print(f"Example input tokens: {input_tokens}...")
        print(f"Example target tokens: {target_tokens}...")
        
        # Verify the shift is correct
        if len(inputs[0]) > 5:
            print(f"\nVerifying shift:")
            for i in range(5):
                input_token = idx2word[inputs[0][i]]
                target_token = idx2word[targets[0][i]]
                expected = idx2word[inputs[0][i+1]] if i+1 < len(inputs[0]) else "N/A"
                print(f"  Position {i}: input='{input_token}' → target='{target_token}' (expected: '{expected}')")
    
    return inputs, targets

class ShakespeareDataset(Dataset):
    """PyTorch Dataset for Shakespeare text"""
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataloaders(inputs, targets, batch_size=32, train_split=0.9, shuffle=True):
    """
    Create train and validation dataloaders.
    
    Args:
        inputs: List of input sequences
        targets: List of target sequences
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        dataset: The full dataset
    """
    dataset = ShakespeareDataset(inputs, targets)
    
    # Split into train and validation if requested
    if train_split < 1.0:
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of train batches: {len(train_loader)}")
        print(f"Number of val batches: {len(val_loader)}")
        
        return train_loader, val_loader, dataset
    
    else:
        # Just training data
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        
        return train_loader, None, dataset

def prepare_shakespeare_data(text_file_or_string, vocab_size=5000, seq_len=150, 
                            batch_size=32, train_split=0.9):
    """
    Complete data preparation pipeline.
    
    Args:
        text_file_or_string: Either a file path or the text string itself
        vocab_size: Maximum vocabulary size
        seq_len: Sequence length for training
        batch_size: Batch size for DataLoader
        train_split: Train/validation split ratio
    
    Returns:
        Dictionary containing all necessary components
    """
    # Load text if it's a file path
    if isinstance(text_file_or_string, str) and text_file_or_string.endswith('.txt'):
        with open(text_file_or_string, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = text_file_or_string
    
    # Step 1: Build vocabulary
    print("Step 1: Building vocabulary...")
    vocab, word2idx, idx2word, tokenizer = build_vocabulary(text, vocab_size)
    
    # Step 2: Create sequences - pass idx2word for proper display
    print(f"\nStep 2: Creating sequences (length={seq_len})...")
    inputs, targets = create_sequences(text, word2idx, idx2word, tokenizer, seq_len)
    
    # Step 3: Create dataloaders
    print(f"\nStep 3: Creating dataloaders (batch_size={batch_size})...")
    train_loader, val_loader, dataset = create_dataloaders(
        inputs, targets, batch_size, train_split
    )
    
    return {
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(vocab),
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'dataset': dataset,
        'seq_len': seq_len
    }


# Add these generation functions to helper_utils.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_tokens(model, prompt_ids, max_length=100, temperature=1.0, 
                   top_k=50, top_p=0.95, repetition_penalty=1.2, 
                   eos_token_id=None, device='cpu'):
    """
    Advanced token generation with multiple sampling strategies.
    
    Args:
        model: The trained model
        prompt_ids: Starting token IDs (list or tensor)
        max_length: Maximum length to generate
        temperature: Controls randomness (0.1=conservative, 2.0=creative)
        top_k: Keep only top k tokens (0=disabled)
        top_p: Nucleus sampling threshold (0.95=default)
        repetition_penalty: Penalty for repeated tokens
        eos_token_id: End of sequence token ID
        device: Device to run on
    
    Returns:
        Generated token IDs as tensor
    """
    model.eval()
    
    # Handle different input formats
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    elif len(prompt_ids.shape) == 1:
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
    else:
        prompt_ids = prompt_ids.to(device)
    
    generated = prompt_ids.clone()
    past_tokens = list(prompt_ids[0].cpu().numpy())
    
    for step in range(max_length - len(prompt_ids[0])):
        # Get model predictions
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logits = model(generated)
        
        # Get the last token's logits
        next_token_logits = logits[0, -1, :].float()
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Penalize all previously generated tokens
            for token_id in set(past_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # Extra penalty for very recent tokens
            if len(past_tokens) > 3:
                for token_id in past_tokens[-3:]:
                    next_token_logits[token_id] /= 1.5
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, len(next_token_logits)))[0][-1]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Apply nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        past_tokens.append(next_token.item())
        
        # Stop if we hit the EOS token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    
    return generated.squeeze(0)

def generate_text(model, prompt, tokenizer, word2idx, idx2word, 
                 max_length=100, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.2, device='cpu'):
    """
    Generate text from a string prompt using advanced sampling.
    
    Args:
        model: Trained model
        prompt: String prompt
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        max_length: Maximum generation length
        temperature: Sampling temperature (0.1-2.0 typical)
        top_k: Top-k filtering (50 is good default)
        top_p: Nucleus sampling (0.95 is good default)
        repetition_penalty: Penalty for repetition (1.2 is good default)
        device: Device to run on
    
    Returns:
        Generated text as string
    """
    # Tokenize prompt
    if not prompt or prompt.isspace():
        # Start with a common word if no prompt
        prompt_tokens = ['the']
    else:
        prompt_tokens = tokenizer(prompt.lower())
    
    # Convert to indices
    prompt_ids = []
    for token in prompt_tokens:
        if token in word2idx:
            prompt_ids.append(word2idx[token])
        else:
            # Try to find similar token
            token_lower = token.lower()
            if token_lower in word2idx:
                prompt_ids.append(word2idx[token_lower])
            else:
                prompt_ids.append(word2idx['<unk>'])
    
    # Ensure we have at least one token
    if not prompt_ids:
        prompt_ids = [word2idx.get('the', word2idx['<unk>'])]
    
    # Generate token IDs
    eos_token_id = word2idx.get('<eos>', None)
    
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        device=device
    )
    
    # Convert to text
    tokens = []
    for idx in generated_ids:
        idx_val = idx.item() if hasattr(idx, 'item') else idx
        token = idx2word.get(idx_val, '<unk>')
        
        # Handle special tokens
        if token == '<nl>' or token == '<newline>':
            tokens.append('\n')
        elif token not in ['<pad>', '<unk>', '<eos>', '<start>']:
            tokens.append(token)
    
    # Join and clean up
    text = ' '.join(tokens)
    
    # Fix punctuation spacing
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    text = text.replace(' \'', '\'').replace('\' ', '\'')
    text = text.replace(' \n ', '\n').replace('\n ', '\n')
    
    return text.strip()

def interactive_generation(model, tokenizer, word2idx, idx2word, device='cpu'):
    """
    Interactive text generation loop.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        device: Device to run on
    """
    print("="*50)
    print("Interactive Shakespeare Text Generation")
    print("="*50)
    print("Enter a prompt to generate text (or 'quit' to exit)")
    print("Commands: 'temp=0.8' to set temperature, 'len=100' to set length")
    print("-"*50)
    
    temperature = 0.8
    max_length = 100
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        # Check for commands
        if prompt.startswith('temp='):
            try:
                temperature = float(prompt[5:])
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid temperature")
                continue
        
        if prompt.startswith('len='):
            try:
                max_length = int(prompt[4:])
                print(f"Max length set to {max_length}")
                continue
            except:
                print("Invalid length")
                continue
        
        # Generate text
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        print("\n" + "="*50)
        print("Generated:")
        print("-"*50)
        print(generated)
        print("="*50)

# Batch generation for multiple prompts
def generate_batch(model, prompts, tokenizer, word2idx, idx2word, 
                  max_length=100, temperature=0.8, device='cpu'):
    """
    Generate text for multiple prompts.
    
    Args:
        model: Trained model
        prompts: List of prompt strings
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        max_length: Maximum generation length
        temperature: Sampling temperature
        device: Device to run on
    
    Returns:
        List of generated texts
    """
    results = []
    for prompt in prompts:
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        results.append(generated)
    return results