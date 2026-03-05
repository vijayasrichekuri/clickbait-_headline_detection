import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

headline = input("Enter a news headline: ")

headline_vector = vectorizer.transform([headline])

prediction = model.predict(headline_vector)
probability = model.predict_proba(headline_vector)

if prediction[0] == 1:
    print("Result: CLICKBAIT")
else:
    print("Result: NOT CLICKBAIT")

print("Confidence:", max(probability[0]))