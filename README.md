# Git 기본 - 상황에 따른 명령어

### 처음으로 (1) 프로젝트를 깃레포지토리로 지정하고 (2) 깃헙에 업로드

- git init
- git add .
- git commit -m "메시지"
- git remote add origin 깃헙주소
- git push origin master

### 다른 레포지토리를 가져올 때

- git clone 다른깃레포지토리주소

### 변경사항을 확인

- git status

### 변경사항 저장

- git add .
- git commit -m "변경사항을 기록하는 메시지"

### 번경사항 깃헙에 업로드

- git push 내가정해준리모트저장소이름 리모트저장소의브랜치

### 브랜치만들고 브랜치로 이동

- git branch 브랜치이름
- git checkout 브랜치이름