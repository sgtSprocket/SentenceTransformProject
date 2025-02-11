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
