import pickle

face_number = 8

# load the evaluation matrix created on find_defect function
with open(f"counter_{face_number}", "rb") as h:
    counter = pickle.load(h)

TP = counter.get("TP")  # true positive
FP = counter.get("FP")  # false positive
FN = counter.get("FN")  # false negative

precision = TP/(TP+FP)
recall = TP / (TP+FN)
F1_score = 2 * (precision*recall) / (precision+recall)

print("Precision score:", precision)
print("Recall score:", recall)
print("F1 score:", F1_score)
