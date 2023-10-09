import os
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import spacy

# Путь к директории с текстовыми файлами
directory_path = 'E:/LynnJustice/2.СборДанных'

# Создайте словарь для хранения слов и их частот
word_count = collections.Counter()
nlp = spacy.load("ru_core_news_sm")

# Обход файлов в директории
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Токенизируйте текст с использованием spaCy
            doc = nlp(text)
            words = [token.text for token in doc]
            # Обновите счетчик слов
            word_count.update(words)

# Создайте словарь слов, где каждому слову присвоен уникальный индекс
word_to_index = {word: index for index, (word, _) in enumerate(word_count.items())}

# Вывод словаря слов
print("Словарь слов:")
print(word_to_index)

# Создайте словарь меток
character_labels = ["Внешность"]
label_to_index = {label: index for index, label in enumerate(character_labels)}


# Функция для преобразования текста в индексы слов
def text_to_indices(text, word_to_index):
    # Токенизируйте текст с использованием spaCy
    doc = nlp(text)
    words = [token.text for token in doc]
    # Преобразуйте каждое слово в индекс, используя словарь word_to_index
    indices = [word_to_index.get(word, -1) for word in words]
    return indices

# Далее идет ваш код для диалогов, который уже создает тензоры для диалогов и меток
dialogs = [
    {"text": "Линн, ты всегда такая красивая", "label": "Внешность"},
]

# Преобразуйте метки в индексы в диалогах
for dialog in dialogs:
    dialog["label"] = label_to_index[dialog["label"]]

# Создайте список числовых представлений диалогов
dialog_indices = [text_to_indices(dialog["text"], word_to_index) for dialog in dialogs]

# Найдите максимальную длину диалога в батче
max_len = max(len(indices) for indices in dialog_indices)

# Выравнивайте длину каждого диалога до максимальной, добавляя нули
padded_dialog_indices = [indices + [0] * (max_len - len(indices)) for indices in dialog_indices]

# Преобразуйте числовые представления в тензоры PyTorch
dialog_tensors = [torch.tensor(indices, dtype=torch.long) for indices in padded_dialog_indices]

# Создайте тензор меток
label_indices = [dialog["label"] for dialog in dialogs]
label_tensor = torch.tensor(label_indices, dtype=torch.long)

# Вывод полученных тензоров
print("Тензоры диалогов:")
for tensor in dialog_tensors:
    print(tensor)

print("Тензор меток:")
print(label_tensor)


# Гиперпараметры модели
vocab_size = len(word_to_index)
embedding_dim = 100
hidden_dim = 128
num_layers = 2
num_classes = len(character_labels)


# Определение RNN модели
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out


# Создание экземпляра модели
model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
print(model)

# Оптимизатор Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Укажите желаемую скорость обучения (lr)

# Определение функции потерь
criterion = nn.CrossEntropyLoss()

# Гиперпараметры обучения
num_epochs = 100


# Обучение модели
def accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


# Обучение модели
for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0

    # Проход по обучающим данным
    for dialog_tensor, label in zip(dialog_tensors, label_tensor):
        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(dialog_tensor.unsqueeze(0))  # добавляем размерность батча

        # Вычисление функции потерь
        loss = criterion(outputs, label.unsqueeze(0))  # добавляем размерность батча

        # Обратное распространение ошибки и оптимизация
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(outputs, label.unsqueeze(0))

    # Выводим среднюю потерю и точность на этой эпохе
    average_loss = total_loss / len(dialog_tensors)
    average_accuracy = total_accuracy / len(dialog_tensors)
    print(f'Эпоха [{epoch + 1}/{num_epochs}], Потеря: {average_loss:.4f}, Точность: {average_accuracy:.4f}')

print('Обучение завершено.')

# Сохранение обученной модели
torch.save(model.state_dict(), 'trained_model.pth')

