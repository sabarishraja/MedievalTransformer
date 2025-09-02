import torch
import numpy as np

class Metrics:
  def __init__(self, number_of_steps=5):
    import evaluate
    self.rouge = evaluate.load("rouge")
    self.bertscore = evaluate.load("bertscore")
    self.accuracy = evaluate.load("accuracy")
    self.number_of_steps = number_of_steps

  def step(self, data, model, style, tokenizer):
    device = next(model.parameters()).device
    style_tensor = torch.full((data.batch_size,), style, dtype=torch.long, device=device)

    # Perplexity on a standard batch
    x, y = data.get_batch('val', device)
    _, loss = model(x, style_tensor, y)
    perplexity = float(torch.exp(loss).item())

    # Generate N tokens and compare to the ground-truth N-shifted slice
    N = data.context_size
    x2, y2 = data.get_batch('val', device, y_shift=N)
    gen_x = model.generate(x2, style_tensor, N)[:, -N:]  # (B, N)

    # Decode for ROUGE / BERTScore
    generated_texts = [tokenizer.decode(i) for i in gen_x.detach().cpu().numpy()]
    reference_texts = [tokenizer.decode(i) for i in y2.detach().cpu().numpy()]

    rouge_results = self.rouge.compute(predictions=generated_texts, references=reference_texts)
    bertscore_results = self.bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")

    # Token accuracy (flatten to 1D lists on CPU)
    preds = gen_x.detach().cpu().numpy().ravel().tolist()
    refs  = y2.detach().cpu().numpy().ravel().tolist()
    accuracy_results = self.accuracy.compute(predictions=preds, references=refs)

    rouge_1 = float(rouge_results["rouge1"])
    rouge_L = float(rouge_results["rougeL"])
    bert_f1 = float(np.mean(bertscore_results["f1"]))
    accuracy = float(accuracy_results["accuracy"])

    return [perplexity, rouge_1, rouge_L, bert_f1, accuracy]

  @torch.no_grad()
  def __call__(self, data, model, style, tokenizer):
    model_was_training = model.training
    model.eval()
    try:
      rows = []
      for _ in range(self.number_of_steps):
        rows.append(self.step(data, model, style, tokenizer))
      agg = np.mean(np.array(rows), axis=0).tolist()
      keys = ["perplexity", "rouge1", "rougeL", "bertscore", "accuracy"]
      return dict(zip(keys, agg))
    finally:
      model.train(model_was_training)
