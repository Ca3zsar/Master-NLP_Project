from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from huggingface_hub import PyTorchModelHubMixin

import torch.nn as nn
# distilbert-base-uncased

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,checkpoint,num_labels, **kwargs): 
        super().__init__()
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained("roberta-base",config=AutoConfig.from_pretrained("roberta-base", output_attentions=True,output_hidden_states=True))
        for param in self.model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Linear(768, num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
