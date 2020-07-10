import Face_Trainer
import Face_Rec
#Face_Trainer.face_trainer()
#Face_Rec.face_rec()
print("Welcome to our program!\n")
looping = True
while (looping):
    choice = input("1)Train Face Recognition\n2)Run Face Recognition\n3)Exit\n")
    if choice == "1":
        Face_Trainer.face_trainer()
    if choice == "2":
        Face_Rec.face_rec()
    if choice == "3":
        looping = False
