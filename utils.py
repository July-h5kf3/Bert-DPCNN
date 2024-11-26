from config import parsers
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader

def read_data(file):
    all_data = open(file,"r",encoding="utf-8").read().split("\n")
    texts,labels = [],[]
    max_length = 0
    for data in all_data:
        if data:
            #print(data)
            text,label = data.split("_separator_")[1],data.split("_separator_")[3]
            #if len(text) > 60:
                #print(text)
             #   continue
            texts.append(text)
            max_length = max(max_length,len(text))
            labels.append(label)
    return texts,labels

class MyDataset(Dataset):
    def __init__(self,texts,labels=None,with_label=True):
        self.all_text = texts
        self.all_label = labels
        self.max_len = parsers().max_len
        self.with_label = with_label
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
        
    def __getitem__(self,index):
        text = self.all_text[index]
        encode_pair = self.tokenizer(text,padding='max_length',truncation = True,max_length = self.max_len,return_tensors = 'pt')
        token_ids = encode_pair['input_ids'].squeeze(0)
        attn_masks = encode_pair['attention_mask'].squeeze(0)
        token_type_ids = encode_pair['token_type_ids'].squeeze(0)
        
        if self.with_label:
            label = int(self.all_label[index])
            return token_ids,attn_masks,token_type_ids,label
        else:
            return token_ids,attn_masks,token_type_ids
    def __len__(self):
        return len(self.all_text)
if __name__ == "__main__":
    train_text, train_label,max_length = read_data("./data/train.txt")
    print(train_text[0], train_label[0],max_length,len(train_text[0]))
    trainDataset = MyDataset(train_text, labels=train_label, with_label=True)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    #for i, batch in enumerate(trainDataloader):
    #    print(batch[0], batch[1], batch[2], batch[3])