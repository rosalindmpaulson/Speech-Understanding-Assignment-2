# -*- coding: utf-8 -*-
"""SU_PA2_Qn1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V0lgQVqmy2kcJgtyctwntdCSReFby9OA
"""

!pip install s3prl torchaudio

!python --version

!git clone https://github.com/microsoft/UniSpeech.git

import shutil
import os
# Define destination
destination_path = "/content/UniSpeech/downstreams/speaker_verification/config/wav2vec2_xlsr.pth"

# Move dataset to /content
shutil.copy('/content/drive/MyDrive/VoxCeleb/wav2vec2_xlsr.pth', destination_path)

# Commented out IPython magic to ensure Python compatibility.
# %cd UniSpeech/downstreams/speaker_verification

!pip install fire

checkpoint_path = 'config/wav2vec2_xlsr.pth'
xx = !python verification.py --model_name wav2vec2_xlsr --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav --checkpoint $checkpoint_path

print(xx)

!wget https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test2.txt

with open('/content/UniSpeech/downstreams/speaker_verification/veri_test2.txt', 'r') as f:
  content = f.read()

contentlist = content.split('\n')
contentlist = [i.split(' ') for i in contentlist]

!ln -s /content/drive/MyDrive/VoxCeleb/vox1 ./vox1
!ln -s /content/drive/MyDrive/VoxCeleb/vox2 ./vox2

print(len(contentlist))

result=[]
originalresult=[]
for ind,i in enumerate(contentlist[:200]):
  print(ind)
  checkpoint_path = 'config/wav2vec2_xlsr.pth'
  wav1path = './vox1/'+i[1]
  wav2path = './vox1/'+i[2]
  originalresult.append(int(i[0]))
  xx = !python verification.py --model_name wav2vec2_xlsr --wav1 $wav1path --wav2 $wav2path --checkpoint $checkpoint_path
  result.append(float(xx[-1]))

result

originalresult

import pandas as pd

df = pd.DataFrame({'originalresult': originalresult, 'result': result})
df['thresholdresult'] = df['result'].apply(lambda x: 1 if x > 0.5 else 0)
df#.to_excel('result.xlsx')

from sklearn.metrics import roc_curve, accuracy_score
import numpy as np

# Extract values
y_true = df['originalresult'].values
y_score = df['result'].values
y_pred = df['thresholdresult'].values

# --- EER Calculation ---
fpr, tpr, thresholds = roc_curve(y_true, y_score)
fnr = 1 - tpr
eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2 * 100  # in %

# --- TAR@1%FAR ---
tar_at_1_fpr_index = np.where(fpr <= 0.01)[0][-1] if np.any(fpr <= 0.01) else 0
tar_at_1_far = tpr[tar_at_1_fpr_index] * 100  # in %

# --- Speaker Identification Accuracy ---
accuracy = accuracy_score(y_true, y_pred) * 100  # in %

# Print results
print(f"EER: {eer:.2f}%")
print(f"TAR @ 1% FAR: {tar_at_1_far:.2f}%")
print(f"Speaker Identification Accuracy: {accuracy:.2f}%")