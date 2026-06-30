# Limitations of the conditional approach

# Limitations of the Machine Learning era

## 1. Machine Learning Cannot Create New Content

Machine learning predicts an output based on existing data.

Example

    Suppose you train a spam classifier.

Input:

    "Congratulations! You won ₹1 lakh."

Output:

    Spam

That's all it does.

It cannot generate a reply such as:

"Thank you for your email. This appears to be spam."

Generative AI can.

## 2. Heavy Dependence on Feature Engineering

In traditional ML, the model doesn't automatically know which information is important.

You often have to manually create useful features.

### House Price Example

Raw data:

    Area
    Bedrooms
    Bathrooms
    Age

You might manually create:

    Price per square foot
    Total rooms
    Age category
    Distance to city center

These engineered features help the model perform better.

This process is called feature engineering, and it requires domain expertise.

Deep learning often learns useful features automatically from raw data.

## 3. Structured Data Works Best

Machine learning performs very well on structured data.

Example:

<img width="843" height="95" alt="image" src="https://github.com/user-attachments/assets/e66eaf5c-9e8d-4a54-b3ef-bc65e4a5e86a" />

However, it struggles more with unstructured data like:

* Books
* Images
* Videos
* Speech
* Long documents

Deep learning models are much better suited to these data types.

4. Poor Understanding of Language

Consider this sentence:

"The bank is on the river."

Traditional ML may simply treat words as independent features (for example, using Bag of Words or TF-IDF), making it difficult to understand that bank refers to the side of a river.

Now consider:

"I deposited money in the bank."

The meaning of bank is different.

Traditional ML has limited ability to capture this context.

Modern language models use context to determine the intended meaning.

5. Difficulty Capturing Long-Term Context

Imagine reading a novel.

Traditional approaches often process sentences independently.

Example:

Sentence 1:
Rahul bought a car.

Sentence 20:
He sold it.

Understanding that He refers to Rahul and it refers to the car requires remembering earlier context.

Traditional ML models generally cannot do this well.

6. Manual Rules Everywhere

Many ML systems require handcrafted rules.

Example:

If salary > 50,000

AND

Experience > 5 years

Approve loan

As systems become larger and more complex, maintaining many rules becomes difficult.

7. Separate Models for Different Tasks

Traditional ML often requires a different model for each problem.

Examples:

Spam detection

↓

Logistic Regression

House price prediction

↓

Linear Regression

Fraud detection

↓

Random Forest

Image classification

↓

CNN

Each task is typically solved with a dedicated model.

Modern foundation models can often perform many tasks with a single model and different prompts.

8. Limited Transfer Learning

Suppose you trained a model to predict house prices.

You cannot simply use that same model to summarize documents.

You need to train a different model.

Large language models can perform many different tasks using the same underlying model.

9. Cannot Have Natural Conversations

Traditional ML systems generally respond with predefined outputs.

Example chatbot:

User:
Hi

Bot:
Hello
User:
How are you?

Bot:
I don't understand.

Modern conversational AI can maintain context and generate natural responses.

10. Data Hunger

Traditional ML often requires:

Clean data
Labeled data
Balanced classes
Significant preprocessing

Without sufficient high-quality data, performance often degrades.

Summary Table
Limitation of Traditional ML	How Modern AI Helps
Cannot generate content	Generates text, images, code, audio, etc.
Requires manual feature engineering	Learns features automatically (deep learning)
Best with structured data	Handles structured and unstructured data
Limited language understanding	Understands context and semantics
Weak long-context handling	Uses attention mechanisms to model long-range dependencies
One model per task	Foundation models support many tasks
Cannot converse naturally	Supports interactive conversations
Often needs lots of labeled data	Modern approaches can leverage self-supervised pretraining



# Limitations of the Neural network

# Generative AI 

A branch of Artificial Intelligence that creates new content rather than just analyzing existing data.

The content can include:

    Text
    Images
    Videos
    Audio
    Music
    Computer code
    3D models

Traditional AI usually predicts or classifies, while Generative AI creates.

<img width="551" height="312" alt="image" src="https://github.com/user-attachments/assets/a0cfb851-7a0d-4a33-9f15-32eb73b21661" />

