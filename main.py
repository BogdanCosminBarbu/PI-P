import cv2
import imutils
from tkinter import *
from tkinter.ttk import Notebook

import numpy as np
from imutils.object_detection import non_max_suppression

#Clasa interfetei cu utilizatorul
class Interface(Frame):

    #Constructor
    def __init__(self):
        super().__init__()

        self.initUI()

    #Functie de configurare a interfetei
    def initUI(self):

        #Titlul
        self.master.title("Prelucrarea Imaginilor")
        self.pack(fill=BOTH, expand=True)

        #Pozitia
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        #Un label ce descrie elementele ce fac parte din interfata
        lbl = Label(self, text="Video-uri")
        lbl.grid(sticky=W, pady=4, padx=5)

        #Notebook-ul care cuprinde fiecare tip de video din interfata + denumiri
        n = Notebook(self)
        f1 = Frame(n)
        f2 = Frame(n)
        f3 = Frame(n)
        f4 = Frame(n)
        n.add(f1, text='Piata Unirii')
        n.add(f2, text='Baza 3')
        n.add(f3, text='Tatarasi')
        n.add(f4, text='Pietonala')
        n.grid(row=1, column=0, columnspan=10, rowspan=10, padx=5, sticky=E+W+S+N)

        #Butoanele care dau start la videoclipul x care este prelucrat pentru fiecare frame din notebook
        hbtn1v2 = Button(f1, text="In timpul noptii", command=lambda: Start_video_prelucrat(switchVideo(5)))
        hbtn1 = Button(f1, text="In timpul zilei", command=lambda: Start_video_prelucrat(switchVideo(3)))

        hbtn2v2 = Button(f2, text="In timpul noptii", command=lambda: Start_video_prelucrat(switchVideo(4)))
        hbtn2 = Button(f2, text="In timpul zilei", command=lambda: Start_video_prelucrat(switchVideo(6)))

        hbtn3 = Button(f3, text="Dashboard Camera", command=lambda: Start_video_prelucrat(switchVideo(14)))

        hbtn4 = Button(f4, text="Pietonala ziua", command=lambda: Start_video_prelucrat(switchVideo(1)))

        #Asezarea butoanelor in fiecare frame
        hbtn1.grid(row=4, column=3, padx=5)
        hbtn1v2.grid(row=4, column=10, padx=5)

        hbtn2.grid(row=4, column=3, padx=5)
        hbtn2v2.grid(row=4, column=10, padx=5)

        hbtn3.grid(row=4, column=3, padx=5)

        hbtn4.grid(row=4, column=3, padx=5)

    #Functie de iesire din aplicatie
    def Exit_App(self):
        self.destroy()
        sys.exit()

#Functia de prelucrare de videouri, de detectie a pietonilor
def Start_video_prelucrat(vid):

    #Initializam descriptorul HOG pentru detectia de pietoni
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #Introducem videoul
    cap = cv2.VideoCapture(vid)

    #Cat timp merge, citim frame cu frame
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            #Dam resize la o alta marime
            image = imutils.resize(image, width=min(1000, image.shape[1]))

            #Detectam pietonii din frame
            (regions, _) = hog.detectMultiScale(image, winStride=(8, 8), padding=(10, 10), scale=1.07)

            #Desenam dreptunghiuri in jurul pietonilor gasiti
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #Aplicam non_max_suppression() la dreptunghiurile gasite aplicand un threshhold
            #pentru a mentine o detectie mai corecta a pietonilor
            regions = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
            pick = non_max_suppression(regions, probs=None, overlapThresh=0.65)

            #Desenam dreptunghiurile finale
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            #Afisam videoul terminat
            cv2.imshow("Video", image)

            #Videoul de inchide daca se apasa pe tasta Q
            cheie = cv2.waitKey(1)
            if cheie == 113:
                break
        else:
            break

    #Inchidere fereastra
    cap.release()
    cv2.destroyAllWindows()


#Functia cu care alegem videoclipul pe care sa il prelucram (switch)
def switchVideo(number):
    switcher = {
        1: "RandomPeople.mp4",
        2: "HowToWalk.mp4",
        3: "PiataUnirii.mp4",
        4: "Baza3Noapte.mp4",
        5: "PiataUniriiNoapte.mp4",
        6: "Baza3.mp4",
        7: "FundatieNoapte.mp4",
        14: "Tatarasi.mp4"
    }
    return switcher.get(number, "Eroare.mp4")
    #in caz de nu este respectat numarul videourilor prezente, se va da play la un video de eroare

#Functia main care porneste interfata grafica
def main():

    root = Tk()
    root.geometry("300x200+300+300")
    app = Interface()
    root.mainloop()

#Start program
if __name__ == '__main__':
    main()
