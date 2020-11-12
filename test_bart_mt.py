from fairseq.models.bart import BARTModel
import torch

model = BARTModel.from_pretrained('pretrained/bart.large', checkpoint_file='model.pt')
model.eval()
model.replace_encoder_embedding_layer()
print(model)

# PATH = 'pretrained/bart.large/custom_model.pt'
# torch.save(model.state_dict(), PATH)
# print('Custom model saved\n')
# model = torch.load(PATH)
# print(model, type(model))
# tokens = model.encode('Hello world!')
# assert tokens.tolist() == [0, 31414, 232, 328, 2]
# print(model.decode(tokens))