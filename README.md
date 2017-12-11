#Logistic regressing
前言:當資料只有兩種結果 Yes or No,簡單的線性回歸無法完全的畫出合適的線。
 ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/1.png)
 
	年紀與購買行為的關係預測。P_Hat表示預測的值
	
 ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/2.png)
 
	取中間值來判斷顧客是否會購買 Y_hat = 0.5

範例: 廣告投放的顧客群選擇 
  ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/3.png)
  
	purchased則表示根據這次廣告的投放後是否購買了產品
 
 ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/4.png)
 
	設定: ID與性別與購買與否無關，所以排除此兩欄位的值


Confusion Matrix:
  ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/5.png)
  
	00:預測沒買實際也沒買 11:預測有買實際也有買 01&10:表示預測錯誤

 ![image](https://github.com/egroeglee/pictures/blob/master/LogisticRegression/6.png)
 
 Output:可以簡單判斷出，年紀大&收入高則購買機率大。
 
上方的斜線稱為[預測邊界]
