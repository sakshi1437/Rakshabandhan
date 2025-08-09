import streamlit as st
import pickle
import string
import os
from nltk.stem.porter import PorterStemmer

# ✅ Custom stopwords set
stopwords_set = {
    'i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",
    'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers',
    'herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if',
    'or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out',
    'on','off','over','under','again','further','then','once','here','there','when','where','why',
    'how','all','any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",
    'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',
    "couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",
    'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",
    'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn'
}

ps = PorterStemmer()

# ✅ Load model & vectorizer safely
MODEL_PATH = "NewBgCmodel.pkl"
VEC_PATH = "Vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    st.error("❌ Model or Vectorizer file missing! Please upload them to the app directory.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# ✅ Text preprocessing
def transform_text(text: str) -> str:
    text = text.lower()
    words = [
        ps.stem(word.strip(string.punctuation))
        for word in text.split()
        if word.strip(string.punctuation).isalnum() and word.lower() not in stopwords_set
    ]
    return " ".join(words)

# ✅ Streamlit UI
st.title("🎁 Rakshabandhan Special Message Classifier")

input_sms = st.text_area("Enter your Rakshabandhan message here:")

if st.button('Classify'):
    if not input_sms.strip():
        st.warning("⚠ Please enter a message to classify.")
    else:
        try:
            # Preprocess input
            transformed_sms = transform_text(input_sms)

            # Vectorize
            vector_input = vectorizer.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

            # Show result according to your label scheme
            if result == 1:
                st.success("💖 Friend Zone Message detected!")
            else:
                st.info("👦 Bhai Zone Message detected!")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
