import asyncio
import sys
import io
import os
import nltk
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, CallbackQuery
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder
import logging
import librosa
import torch
import torch.nn.functional as f
from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
sampling_rate = processor.feature_extractor.sampling_rate
model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
API_TOKEN = '7260511059:AAFfHvH7AOnBuc3ByUS5LX2f2Gu2ScVZ8ig'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
reka = Recognizer()
# nltk.download('punkt')
data = [("Я очень беспокоюсь о своих финансах", "беспокойство"),
    ("Мне спокойно, я все понимаю", "спокойствие"),
    ("Я не знаю, как решить эту проблему", "беспокойство"),
    ("Все будет хорошо, я уверен в этом", "спокойствие"),
    ("Я чувствую тревогу из-за кредита", "беспокойство"),
    ("Я спокоен и готов к новым вызовам", "спокойствие"),
    ("У меня много вопросов, и я переживаю", "беспокойство"),
    ("Я чувствую себя комфортно, когда общаюсь с вами", "спокойствие")]
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
model_hearts_of_iron = make_pipeline(CountVectorizer(), MultinomialNB())
model_hearts_of_iron.fit(X_train, y_train)
print('Бот Активирован')


def predict(audio_buffer):
    speech, sr = librosa.load(io.BytesIO(audio_buffer), sr=sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    scores = f.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return [{"label": config.id2label[i], "score": round(score, 5)} for i, score in enumerate(scores)]


def predict_mood(text):
    prediction = model_hearts_of_iron.predict([text])
    return prediction[0]


@dp.message(Command(commands=['start']))
async def start(message: types.Message):
    await message.answer(f'Здравствуй, {message.from_user.first_name}. Отправь мне аудио или текст, чтобы по нему я смог определить эмоцию человека')
@dp.message()
async def handle_audio_voice(message: types.Message):
    if message.content_type not in [types.ContentType.AUDIO, types.ContentType.VOICE, types.ContentType.TEXT]:
        await message.answer('Отправь мне аудио или текст, чтобы по нему я смог определить эмоцию человека')
        return
    if message.text:
        moodila = predict_mood(message.text.strip())
        await message.answer(f'Человек чувствует {moodila}')
        return
    elif message.audio:
        file_id = message.audio.file_id
        file_name = message.audio.file_unique_id + '.ogg'
    else:
        file_id = message.voice.file_id
        file_name = message.voice.file_unique_id + '.ogg'
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    file_path = os.path.join('C:/users/user/PycharmProjects/TechnologiaEtTriumphum/.venv/audio/' + file_name)
    with open(file_path, 'wb') as govno:
        govno.write(downloaded_file.getvalue())
    but1 = InlineKeyboardButton(text='По интонации', callback_data='1' + file_name)
    but2 = InlineKeyboardButton(text='По тексту', callback_data='2' + file_name)
    await message.answer('Выбери способ определения эмоции в сообщении:', reply_markup=InlineKeyboardMarkup(inline_keyboard=[[but1], [but2]]))


@dp.callback_query()
async def callbacks_handler(callback: CallbackQuery):
    dermo = callback.data
    if dermo[0] == '1':
        file_put = 'C:/users/user/PycharmProjects/TechnologiaEtTriumphum/.venv/audio/' + dermo[1:]
        with open(file_put, 'rb') as audio18:
            fakel = audio18.read()
            audio_buffer = io.BytesIO(fakel)
            predictions: list = predict(audio_buffer.getvalue())
            pred = max(predictions, key=lambda x: x["score"])["label"]
            match pred:
                case 'neutral':
                    emotion = 'спокоен (нейтрален)'
                case 'sad':
                    emotion = 'беспокоен (опечален)'
                case 'angry':
                    emotion = 'беспокоен (зол)'
                case 'positive':
                    emotion = 'спокоен (счастлив)'
                case 'others':
                    emotion = 'ощущает другие эмоции'
            await callback.message.answer(f'Человек {emotion}')
    elif dermo[0] == '2':
        file_put = 'C:/users/user/PycharmProjects/TechnologiaEtTriumphum/.venv/audio/' + dermo[1:]
        audio = AudioSegment.from_file(file_put, 'ogg')
        audio.export(file_put, format='wav')
        with AudioFile(file_put) as source:
            audio_data = reka.record(source)
        try:
            textovik = reka.recognize_google(audio_data, language='ru-RU')
        except UnknownValueError:
            await callback.message.answer('Не удалось распознать текст, попробуйте ещё раз')
        except RequestError:
            await callback.message.answer('Не удалось распознать текст, попробуйте ещё раз')
        moodila = predict_mood(textovik)
        await callback.message.answer(f'Человек чувствует {moodila}')


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
