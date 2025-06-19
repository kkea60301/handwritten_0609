參考此出處:
https://steam.oxxostudio.tw/category/python/ai/ai-number-recognizer.html
https://syshen.medium.com/%E6%94%B9%E5%96%84-cnn-%E8%BE%A8%E8%AD%98%E7%8E%87-dac9fce59b63
https://github.com/syshen/mnist-cnn/blob/master/mnist-CNN-datagen.ipynb?source=post_page-----dac9fce59b63---------------------------------------

多位數:
https://github.com/shaohua0116/MultiDigitMNIST
https://github.com/shakibyzn/two-digit-mnist-recognition


#做完修改執行以下推送
git remote -v
git pull
git add .
git commit -m "訊息"
git push origin main
git push

#電腦已有專案資料夾，同步指令
git init
git remote add origin <GitHub儲存庫URL>
git fetch origin
git reset --hard origin/main #本地亂改一通，發現都不要了，只想要跟 GitHub 上一樣，就用這個指令
git reset --soft origin/main #發現 commit 做錯了，想重做一次，但又不想把所有修改都刪掉，可以用這個指令。
git pull
git pull origin main

#情況二：A 電腦需要重新下載完整專案
git clone https://github.com/你的帳號/你的repo.git
git fetch origin
git reset --hard origin/main