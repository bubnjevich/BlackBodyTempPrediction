import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageTk
from cnn.cnn import cnn_prediction

from elastic_net.elastic_net import *
def start_prediction():
	file = entry.get()
	if file:
		if method_var.get() == "CNN":
			predict_by_cnn(file)
		else:
			predict_by_elastic_net(file)
	else:
		messagebox.showwarning("Warning", "Choose file")


def predict_by_elastic_net(file):
	try:
		Y_pred, T = predict(file)
		image = Image.open(f"results_net/{Y_pred:.2f}K.png")
		image = image.resize((400, 300))
		image_tk = ImageTk.PhotoImage(image)
		canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
		canvas.image = image_tk
		label_temperature.config(text=f"Predicted Temperature: {Y_pred:.2f}K. Real Temperature: {T:.2f}K.")
	except Exception as e:
		messagebox.showerror("Error", str(e))


def predict_by_cnn(file):
	try:
		Y_pred, T = cnn_prediction("model/planck.h5", file)
		image = Image.open(f"results_cnn/{Y_pred:.2f}K.png")
		image = image.resize((400, 300))
		image_tk = ImageTk.PhotoImage(image)
		canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
		canvas.image = image_tk
		label_temperature.config(text=f"Predicted Temperature: {Y_pred:.2f}K. Real Temperature: {T:.2f}K.")
	except Exception as e:
		messagebox.showerror("Error", str(e))


def clear_fields():
	entry.delete(0, tk.END)
	label_temperature.config(text="")
	canvas.delete("all")

def choose_file():
	file_path = filedialog.askopenfilename()
	if file_path:
		entry.delete(0, tk.END)
		entry.insert(0, file_path)


# Kreiranje Tkinter prozora
if __name__ == '__main__':
	
	window = tk.Tk()
	window.title("Temperature prediction")
	window.geometry("700x500")
	
	label_file = tk.Label(window, text="Choose file:")
	label_file.grid(row=0, column=0, padx=10, pady=10)
	
	entry = tk.Entry(window, width=30)
	entry.grid(row=0, column=1, padx=10, pady=10)
	
	button_file = tk.Button(window, text="Choose", command=choose_file)
	button_file.grid(row=0, column=2, padx=10, pady=10)
	
	
	method_var = tk.StringVar()
	method_var.set("CNN")
	label_method = tk.Label(window, text="Method:")
	label_method.grid(row=1, column=0, padx=10, pady=10)
	
	radio_cnn = tk.Radiobutton(window, text="CNN", variable=method_var, value="CNN")
	radio_cnn.grid(row=1, column=1, padx=10, pady=10)
	
	radio_en = tk.Radiobutton(window, text="Elastic Net", variable=method_var, value="ElasticNet")
	radio_en.grid(row=1, column=2, padx=10, pady=10)
	
	button_start = tk.Button(window, text="Start", command=start_prediction)
	button_start.grid(row=2, column=0, padx=10, pady=10)
	
	button_clear = tk.Button(window, text="Clear", command=clear_fields)
	button_clear.grid(row=2, column=1, padx=10, pady=10)
	
	label_temperature = tk.Label(window)
	label_temperature.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
	
	canvas = tk.Canvas(window, width=400, height=300)
	canvas.grid(row=3, column=0, columnspan=3, padx=10, pady=10)
	
	window.mainloop()