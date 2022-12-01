from transformers import BertTokenizer, BertForMaskedLM
import torch

model_path = './alephbertgimmel-small/ckpt_29400--Max128Seq'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# if not finetuning - disable dropout
model.eval()

sentence = 'דני הלך לבית [MASK] היום.'

output = model(tokenizer.encode(sentence, return_tensors='pt'))
# the [MASK] is the 4th token (including [CLS])
top_5 = torch.topk(output.logits[0, 4, :], 5)[1]
print('\n'.join(tokenizer.convert_ids_to_tokens(top_5))) # should print ספר / הספר / כנסת