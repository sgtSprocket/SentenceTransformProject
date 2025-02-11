Sentence Transformer Project for Fetch, by William G

Runs on Python 3.10.11, using PyCharm 2022.2.5

Due to issues with connecting Github to PyCharm, as well as file size issues, no dependencies are provided.
In order to run main.py, you will need the following libraries:
-  sentence-transformers 3.4.1
-  scikit-learn 1.6.1
-  pyTorch 2.6.0
-  numpy 2.2.2

Explanations of my design choices and ideas are written in the comments within the code.

NOTE: Due to time constraints, I was only able to complete Task 2A, and not Task 2B, but I can explain how I would have approached Task 2B:

As this assessment was my first time working with Sentence Transformers, my immediate instinct was to hunt down as much documentation as I could and see what
kinds of functionality the framework already offered, and its a very robust framework with many best practices already established. This guide here looks to
be a good start for Sentiment Analysis, for example: https://huggingface.co/blog/sentiment-analysis-python. We also live in an age of AI assistants being
widely available, but since it can be difficult to validate where exactly AI chat bots are pulling their information from (or if they're just making things up
and hallucinating), I usually prefer hunting down original documentation first.


TASK 3: Training Considerations

Here I will answer the hypothetical questions presented in the assessments:

Training the Model:

1- Should the entire network be frozen?
  Freezing the entire network would stop all training/learning completely. I can see this being useful if there is a fixed state you want the model to maintain
  while using the model to generate new outputs. This is likely very beneficial for a ML product that is made publicly available: it makes the behavior of the model
  more predictable and controllable, and prevents bad actors from "poisoning" your model by making it learn weird inputs/outputs. Any scenario where the model would
  be exposed to unstrustworthy inputs (anything remotely public) I would freeze it.

2- If only the transformer backbone is frozen.
  If the transformer is frozen, your embedded sentences will be the same every time. I'm still fairly new to Sentence Transformers, but from my understanding,
  the embeds of the sentences are the computer's way of "understanding" the sentences. If this is part of the model is frozen, the model will never learn
  new encoding patterns and new relationships between words, but can still learn new weights/biases for the purposes of performing tasks on new inputs.
  I imagine this is useful for controlling what specific patterns and relationships within sentences the model will understand.

3- Freezing task-specific elements.
  For the opposite of freezing just the transformer, you now have a model that can learn new embedding patterns but will no longer learn for applied tasks.
  In this instance, the model will be able to gain more understanding of language structure as you can feed more sentences to the encoder, while all other
  tasks remain at the level that they were. A big use case I can see for this is if you decide to turn your transformer into a pre-trained model for distribution,
  in which case developers will set up their extraneous tasks on their own anyways, so you only care about building the transformer and nothing else.


Transfer Learning:

Since I am familiar with Transfer Learning in the context of image classification, I can translate some of those ideas over to language processing,
since approaching sentences and text as "1D images" has proven useful so far:

1- Choosing a pre-trained model.
  I would choose a robust and general-purpose NLP model. Something that's been trained on a lot of data, and a high variety of data.

2- Layers to freeze or unfreeze?
  Freeze the early layers, unfreeze the middle and later layers. I would say start with maybe the first 25% of the model frozen and the rest unfrozen.

3- The rationale.
  The benefit of transfer learning is that you start with a trained and functional model of the bat. According to my research, a popular transfer learning strategy
  in language processing is to take a "broad" model and train it to focus on more specific topics. We can do this with the freezing technique I mentioned in point 2.
  As you progress through the layers of the model, patterns learned by the model gradually shift from general to specific. So, if you want to steer the model towards
  learning for a specific tasks, you really want to focus on updating the later layers.


Task 4: Training Loop

I did my own training for the sentence classification portion of Task 2A. Granted, due to limited data, it isn't very robust, but it's functional.
The idea was pretty straightforward: We have 2 classes of data, one for sentences talking about pistachio nuts, the other for sentences talking about trucks.
They are labelled and then packaged as a data loader. The actual design of the network is VERY fast and loose and I know for a fact it isn't optimal, but given
that this is my first time working with natural language processing, the limited training data, as well as the time limit, I kept it simple.
Learning rate is high due to low epoch count and a lack of batching/sampling. This will result in extreme overfitting, but given that the input data was designed
to be as "black and white" as possible, the model is still functional.


My approach to the assessment was very fast and loose, and I understand the end result is a little messy and incomplete, but as stated before this is my first
time working with NLP and sentence transformers and I learned a lot! I was able to take my academic knowledge of the image classification CNN's I worked on and
apply those ideas to a new machine learning task and this was definitely a valuable experience for me. Thank you for taking the time to read this wall of text!
