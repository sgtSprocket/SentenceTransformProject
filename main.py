import sentence_transformers as st
import torch.utils.data as tud
import torch
import torch.nn as nn

# To keep things quick and simple, I'm not seperating things into their own python files, so just main.py
# Running on Python 3.10.11, via PyCharm

class network(nn.Module):
    def __init__(self):
        super().__init__()

        # Very rough and weird NN setup due to time limitations, but I can confirm 2 things:
        # First, we only have 1 input channel as the embeds are fixed-length vectors (384 numbers), which
        # we can use linear to reduce down to a smaller feature set. 64 is arbitrary, but its a nice smaller
        # number to work with
        # Second, this is my very first time exploring sentence transformers! I am used to convolutional neural
        # networks in the context of 2D images, but from my research it appears that CNN's are also great for
        # text-based stuff, which makes sense: Images are basically just 2D matrices where you want to learn
        # relationships between pixels, and sentences are just the 1-dimensional version of that idea.
        self.linear = nn.Linear(384, 64)
        self.conv1d = nn.Conv1d(1, 1, kernel_size = 32)
        self.maxpool = nn.MaxPool1d(kernel_size=16)
        self.conv1d2 = nn.Conv1d(1, 1, kernel_size=8)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.conv1d(x)
        x = self.maxpool(x)
        #x = self.conv1d2(x)
        #x = self.relu(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    # Setup a pretrained model
    # Model pulled from this list: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    # I chose minilm-l12 because it seems to have a nice balance of size and speed
    model = st.SentenceTransformer("all-MiniLM-L12-v2")

    # some sentences, very obviously split into 2 categories: pistachio nuts and pickup trucks
    sentences = [
        "Yeah I love pistachio nuts!",
        "Did you eat all of the pistachios?",
        "I bought some pistachios at the store today.",
        "I ran out of pistachios...",
        "I need to eat more pistachios, I can't help myself.",
        "I bought a new pickup truck last week.",
        "Did you lift your truck?",
        "My truck broke down again!",
        "One time when I was a kid, I accidentally set my dad's truck on fire",
        "And now the truck is gone."
    ]

    # labels for the data, 0 means pistachio, 1 means truck
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # turning our disgusting human sentences into pretty numbers
    embeds = model.encode(sentences)

    t_embeds = torch.empty(0)

    for a in embeds:
        t_embeds = torch.cat((t_embeds, torch.from_numpy(a)))

    t_embeds = torch.reshape(t_embeds, (len(labels), 384))

    #print(t_embeds)
    #print(t_embeds.shape)

    # turn data into a training dataloader
    t_labels = torch.tensor(labels)
    # t_labels = t_labels.to(torch.float32)
    embeds = torch.from_numpy(embeds)
    dataset = tud.TensorDataset(t_embeds, t_labels)
    dataloader = tud.DataLoader(dataset)

    # establish our neural network
    myNN = network()

    # the following code is standard pytorch model usage
    # I'm falling back onto what I know from school, which in turn follows pyTorch's official documentation
    # See here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # For time constraint reasons, I'm following this documentation quite closely as a template, keeping things simple
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(myNN.parameters(), lr=0.1)

    # training time!
    # a very small training set, so this model will probably not be that great
    for epoch in range(1000):
        for iter, a in enumerate(dataloader):
            sentences, labels = a

            #print(sentences)

            optimizer.zero_grad()

            results = myNN(sentences)

            loss = criterion(results, labels)
            loss.backward()
            optimizer.step()

            #if(iter % 10 == 8):
                #print(loss)


    # let's make a new sentence, embed it, and feed it
    sentence = ["I'm gonna go eat some pistachios, want to join me?"]
    embed = model.encode(sentence)
    embed = torch.from_numpy(embed)

    # Let's see how the model interprets this sentence about pistachios
    # The model should be extremely overfitted due to small sample size + a lot of epochs, so its answer will either
    # be extreme, or broken, with no nuance in-between
    # The first value is the confidence in pistachio, the second is confidence in pickup truck
    result = myNN(embed)
    print(result)

