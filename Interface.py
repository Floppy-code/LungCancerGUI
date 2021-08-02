import os
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename

#Plaid-ml
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from Manager import Manager

class Interface:
    def __init__(self, master=None):
        #My variables
        self.manager = Manager()
        self.latest_treeview_slice = None
        self.debug_output = True
        self.segmentation_cbtn_state = tk.IntVar()
        self.loc_segmentation_cbtn_state = tk.IntVar()
        self.localization_dict = {}
        
        self.localization_folder = './localization'
        self.localization_mode = self.manager.get_localization()
        self.load_localization_dictionary()

        #Tkinter GUI
        self.main_frame = ttk.Frame(master)

        self.model_functions_frame = ttk.Labelframe(self.main_frame)

        self.load_model_btn = ttk.Button(self.model_functions_frame)
        self.load_model_btn.configure(text = self.localization_dict['load_model_btn'], width='20')
        self.load_model_btn.place(anchor='nw', x='5', y='0')
        self.load_model_btn.configure(command=self.load_model_btn_pressed)
        self.predict_model_btn = ttk.Button(self.model_functions_frame)
        self.predict_model_btn.configure(text=self.localization_dict['predict_model_btn'], width='20')
        self.predict_model_btn.place(anchor='nw', x='5', y='30')
        self.predict_model_btn.configure(command=self.predict_model_btn_pressed)
        
        self.model_functions_frame.configure(height='80', text=self.localization_dict['model_functions_frame'], width='145')
        self.model_functions_frame.place(anchor='nw', x='10', y='0')
        self.dataset_functions_frame = ttk.Labelframe(self.main_frame)
        self.load_ctscan_btn = ttk.Button(self.dataset_functions_frame)
        self.load_ctscan_btn.configure(text=self.localization_dict['load_ctscan_btn'], width='20')
        self.load_ctscan_btn.place(anchor='nw', x='5', y='0')
        self.load_ctscan_btn.configure(command=self.load_ct_scan_btn_pressed)
        self.load_ctsegment_btn = ttk.Button(self.dataset_functions_frame)
        self.load_ctsegment_btn.configure(text=self.localization_dict['load_ctsegment_btn'], width='20')
        self.load_ctsegment_btn.place(anchor='nw', x='5', y='30')
        self.load_ctsegment_btn.configure(command=self.load_ct_segment_btn_pressed)
        self.dataset_functions_frame.configure(height='80', text=self.localization_dict['dataset_functions_frame'], width='145')
        self.dataset_functions_frame.place(anchor='nw', x='160', y='0')
                
        self.info_frame = ttk.Labelframe(self.main_frame)
        self.label4 = ttk.Label(self.info_frame)
        self.label4.configure(text=self.localization_dict['label4'], width='17')
        self.label4.grid(column='0', padx='5', pady='0', row='0')
        self.label4.rowconfigure('0', minsize='0', weight='0')
        self.label4.columnconfigure('0', minsize='0')
        self.label5 = ttk.Label(self.info_frame)
        self.label5.configure(text=self.localization_dict['label5'], width='17')
        self.label5.grid(column='0', padx='5', pady='2', row='1')
        self.label5.columnconfigure('0', minsize='0')
        self.dataset_res_lbl = ttk.Label(self.info_frame)
        self.dataset_res_lbl.configure(text=self.localization_dict['dataset_res_lbl'], width='17')
        self.dataset_res_lbl.grid(column='1', row='0')
        self.input_shape_lbl = ttk.Label(self.info_frame)
        self.input_shape_lbl.configure(text=self.localization_dict['input_shape_lbl'], width='17')
        self.input_shape_lbl.grid(column='1', row='1')
        self.label10 = ttk.Label(self.info_frame)
        self.label10.configure(text=self.localization_dict['label10'], width='17')
        self.label10.grid(column='2', row='0')
        self.label11 = ttk.Label(self.info_frame)
        self.label11.configure(text=self.localization_dict['label11'], width='17')
        self.label11.grid(column='0', row='2')
        self.label12 = ttk.Label(self.info_frame)
        self.label12.configure(width='17')
        self.label12.grid(column='2', row='2')
        self.patient_filename_lbl = ttk.Label(self.info_frame)
        self.patient_filename_lbl.configure(text=self.localization_dict['patient_filename_lbl'], width='17')
        self.patient_filename_lbl.grid(column='3', row='0')
        self.slice_count_lbl = ttk.Label(self.info_frame)
        self.slice_count_lbl.configure(text=self.localization_dict['slice_count_lbl'], width='17')
        self.slice_count_lbl.grid(column='1', row='2')
        self.info_frame.configure(text=self.localization_dict['info_frame'])
        self.info_frame.place(anchor='nw', height='80', width='480', x='310', y='0')

        self.original_img_frame = ttk.Labelframe(self.main_frame)
        self.original_img_frame.configure(height='200', text=self.localization_dict['original_img_frame'], width='200')
        self.original_img_frame.place(anchor='nw', height='240', width='240', x='415', y='90')
        self.cnn_img_frame = ttk.Labelframe(self.main_frame)
        self.cnn_img_frame.configure(height='200', text=self.localization_dict['cnn_img_frame'], width='200')
        self.cnn_img_frame.place(anchor='nw', height='240', width='240', x='415', y='330')
        self.frame6 = ttk.Frame(self.main_frame)
        self.label15 = ttk.Label(self.frame6)
        self.label15.configure(text='Juraj Oberta | 2021 | Fakulta Riadenia a Informatiky')
        self.label15.grid(column='0', padx='5', pady='2', row='0', sticky='w')
        self.frame6.configure(height='200', relief='groove', width='200')
        self.frame6.place(anchor='nw', height='20', width='800', x='0', y='580')
        
        self.tree_frame = ttk.Frame(self.main_frame)
        self.tree_frame.place(anchor='nw', height='480', width='395', x='10', y='90')

        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side = tk.RIGHT, fill = tk.Y)

        self.slice_treeview = ttk.Treeview(self.tree_frame, columns=('Slice', 'Global Y position', 'Probability'), yscrollcommand = self.tree_scroll.set, selectmode = 'browse')
        self.slice_treeview.heading('#0', text = '',anchor=tk.W)
        self.slice_treeview.heading('#1', text = self.localization_dict['slice_treeview1'],anchor=tk.W)
        self.slice_treeview.heading('#2', text = self.localization_dict['slice_treeview2'],anchor=tk.W)
        self.slice_treeview.heading('#3', text = self.localization_dict['slice_treeview3'],anchor=tk.W)

        self.slice_treeview.column('#0', stretch = tk.NO, width = 0)
        self.slice_treeview.column('#1', stretch = tk.YES, width = '126')
        self.slice_treeview.column('#2', stretch = tk.YES, width = '126')
        self.slice_treeview.column('#3', stretch = tk.YES, width = '125')

        self.slice_treeview.bind('<<TreeviewSelect>>', self.slice_treeview_selected)

        self.tree_scroll.config(command = self.slice_treeview.yview)

        self.slice_treeview.place(anchor='nw', height='480', width='379', x='0', y='0')

        self.options_frame = ttk.Labelframe(self.main_frame)
        self.segmentation_cbtn = ttk.Checkbutton(self.options_frame, variable = self.segmentation_cbtn_state)
        self.segmentation_cbtn.configure(text=self.localization_dict['segmentation_cbtn'], width='16')
        self.segmentation_cbtn.pack(side='top')
        self.input_cbx = ttk.Combobox(self.options_frame)
        self.input_cbx.pack(padx='5', pady='2', side='top')
        self.input_cbx['values'] = (self.localization_dict['input_cbx1'], 'VGG19', 'ResNet50', 'DenseNet169')
        self.input_cbx.current(0)
        self.reshape_btn = ttk.Button(self.options_frame)
        self.reshape_btn.configure(text=self.localization_dict['reshape_btn'])
        self.reshape_btn.pack(fill='x', padx='5', pady='2', side='top')
        self.reshape_btn.configure(command=self.reshape_btn_pressed)
        self.show_original_btn = ttk.Button(self.options_frame)
        self.show_original_btn.configure(text=self.localization_dict['show_original_btn'])
        self.show_original_btn.pack(fill='x', padx='5', pady='2', side='top')
        self.show_original_btn.configure(command=self.show_original_btn_pressed)
        self.show_cnn_btn = ttk.Button(self.options_frame)
        self.show_cnn_btn.configure(text=self.localization_dict['show_cnn_btn'])
        self.show_cnn_btn.pack(fill='x', padx='5', pady='2', side='top')
        self.show_cnn_btn.configure(command=self.show_cnn_btn_pressed)
        self.options_frame.configure(height='200', text=self.localization_dict['options_frame'], width='200')
        self.options_frame.place(anchor='nw', height='155', width='125', x='665', y='90')

        self.language_cbox = ttk.Combobox(self.main_frame)
        self.language_cbox.place(anchor='nw', width='125', x='665', y='255')
        self.language_cbox['values'] = ('English', 'Slovensky')
        self.language_cbox.bind('<<ComboboxSelected>>', self.language_updated)
        if self.localization_mode == 'en':
            self.language_cbox.current(0)
        else:
            self.language_cbox.current(1)

        self.main_frame.configure(height='600', width='800')
        self.main_frame.grid(column='0', row='0')

        # Main widget
        self.mainwindow = self.main_frame

    def language_updated(self, event):
        if self.language_cbox.get() == 'English' and self.localization_mode == 'en':
            return
        elif self.language_cbox.get() == 'Slovensky' and self.localization_mode == 'sk':
            return
        else:
            if self.language_cbox.get() == 'English':
                self.localization_mode = 'en'
            if self.language_cbox.get() == 'Slovensky':
                self.localization_mode = 'sk'
            self.manager.set_localization(self.localization_mode)
            self.load_localization_dictionary()

            #Reload all texts
            self.load_model_btn.configure(text = self.localization_dict['load_model_btn'], width='20')
            self.predict_model_btn.configure(text=self.localization_dict['predict_model_btn'], width='20')
            self.model_functions_frame.configure(height='80', text=self.localization_dict['model_functions_frame'], width='145')
            self.load_ctscan_btn.configure(text=self.localization_dict['load_ctscan_btn'], width='20')
            self.load_ctsegment_btn.configure(text=self.localization_dict['load_ctsegment_btn'], width='20')
            self.dataset_functions_frame.configure(height='80', text=self.localization_dict['dataset_functions_frame'], width='145')
            self.label4.configure(text=self.localization_dict['label4'], width='17')
            self.label5.configure(text=self.localization_dict['label5'], width='17')
            self.label10.configure(text=self.localization_dict['label10'], width='17')
            self.label11.configure(text=self.localization_dict['label11'], width='17')
            self.info_frame.configure(text=self.localization_dict['info_frame'])
            self.original_img_frame.configure(height='200', text=self.localization_dict['original_img_frame'], width='200')
            self.cnn_img_frame.configure(height='200', text=self.localization_dict['cnn_img_frame'], width='200')
            self.slice_treeview.heading('#0', text = '',anchor=tk.W)
            self.slice_treeview.heading('#1', text = self.localization_dict['slice_treeview1'],anchor=tk.W)
            self.slice_treeview.heading('#2', text = self.localization_dict['slice_treeview2'],anchor=tk.W)
            self.slice_treeview.heading('#3', text = self.localization_dict['slice_treeview3'],anchor=tk.W)
            self.segmentation_cbtn.configure(text=self.localization_dict['segmentation_cbtn'], width='16')
            self.input_cbx['values'] = (self.localization_dict['input_cbx1'], 'VGG19', 'ResNet50', 'DenseNet169')
            self.reshape_btn.configure(text=self.localization_dict['reshape_btn'])
            self.show_original_btn.configure(text=self.localization_dict['show_original_btn'])
            self.show_cnn_btn.configure(text=self.localization_dict['show_cnn_btn'])
            self.options_frame.configure(height='200', text=self.localization_dict['options_frame'], width='200')


    def slice_treeview_selected(self, event):
        selected_item = event.widget.selection()
        for item in selected_item:
            self.latest_treeview_slice = self.slice_treeview.item(item)['values'][0]
        self.show_ct_image(0, self.latest_treeview_slice)
        if (self.manager.is_detection_dataset_reshaped()):
            self.show_ct_image(1, self.latest_treeview_slice)

    def show_ct_image(self, type = 0, slice_no = 0):
        #Types: 0 - Default CT image  1 - CNN input image
        image_array = self.manager.get_slice_image(type, slice_no, (220, 220))
        if (type == 0):
            image = ImageTk.PhotoImage(image = Image.fromarray(image_array))
            image_label = ttk.Label(self.original_img_frame)
            image_label.image = image
            image_label.configure(image = image)
            image_label.place(anchor='nw', height='220', width='220', x='0', y='0')
        elif (type == 1):
            image = ImageTk.PhotoImage(image = Image.fromarray(image_array))
            image_label = ttk.Label(self.cnn_img_frame)
            image_label.image = image
            image_label.configure(image = image)
            image_label.place(anchor='nw', height='220', width='220', x='0', y='0')
        else:
            raise Exception("Not implemented yet!")


    def load_model_btn_pressed(self):
        filename = askopenfilename()
        try:
            self.manager.load_detection_network(filename)
        except:
            self.debug(self.localization_dict['load_model_err'])
            return
        shape = self.manager.get_network_input_shape(0)
        self.input_shape_lbl.configure(text = "({}, {}, {})".format(shape[1], shape[2], shape[3]), width = '17')

    def predict_model_btn_pressed(self):
        try:
            self.manager.predict_detection_model()
            self.refresh_treeview()
        except:
            self.debug(self.localization_dict['predict_err'])

    def load_ct_scan_btn_pressed(self):
        filename = askopenfilename()
        try:
            self.manager.load_patient_data(filename)
        except:
            self.debug(self.localization_dict['load_ct_scan_err'])
            return

        resolution = self.manager.get_dataset_resolution()
        self.slice_treeview.delete(*self.slice_treeview.get_children())
        self.patient_filename_lbl.config(text = filename.split('/')[len(filename.split('/')) - 1])
        self.dataset_res_lbl.config(text = "{}x{} px".format(resolution[1], resolution[2]), width = '17')
        self.slice_count_lbl.config(text = resolution[0])
        for slice in self.manager.get_treeview_data():
            self.slice_treeview.insert('', 'end', iid = slice[0], text = slice[0], values = (slice[0], slice[1], slice[2]))

    def refresh_treeview(self):
        self.slice_treeview.delete(*self.slice_treeview.get_children())
        for slice in self.manager.get_treeview_data():
            probability = slice[2] * 100.0
            if probability >= 50.0:
                probability = "Y, {0:.2f}%".format(probability)
            else:
                probability = "N, {0:.2f}%".format(probability)
            self.slice_treeview.insert('', 'end', iid = slice[0], text = slice[0], values = (slice[0], slice[1], probability))

    def load_ct_segment_btn_pressed(self):
        filename = askopenfilename()
        try:
            self.manager.load_segmentation_data(filename)
        except:
            self.debug(self.localization_dict['load_ct_segment_err'])

    def reshape_btn_pressed(self):
        if self.manager.is_dataset_loaded() and self.manager.is_detection_network_loaded():
            self.manager.reshape_for_prediction((self.segmentation_cbtn_state.get() == 1), self.input_cbx.get().lower())
        else:
            self.debug(self.localization_dict['reshape_err'])

    def show_original_btn_pressed(self):
        if self.latest_treeview_slice != None and self.manager.is_dataset_loaded():
            image = self.manager.get_slice_image(0, self.latest_treeview_slice, 0)
            plt.figure(figsize = (8, 8))
            plt.imshow(image, cmap = 'gray')
            plt.show()
        else:
            self.debug(self.localization_dict['show_original_err'])

    def show_cnn_btn_pressed(self):
        if self.latest_treeview_slice != None and self.manager.is_detection_dataset_reshaped():
            image = self.manager.get_slice_image(1, self.latest_treeview_slice, 0)
            if image.shape[2] == 1:
                image = np.squeeze(image)
            plt.figure(figsize = (8, 8))
            plt.imshow(image, cmap = 'gray')
            plt.show()
        else:
            self.debug(self.localization_dict['show_cnn_err'])

    def load_localization_btn_pressed(self):
        filename = askopenfilename()
        try:
            self.manager.load_localization_network(filename)
        except:
            self.debug(self.localization_dict['load_localization_err'])
            return
        shape = self.manager.get_network_input_shape(1)
        self.localization_shape_lbl.config(text = "({}, {}, {})".format(shape[1], shape[2], shape[3]), width = '17')

    def predict_localization_btn_pressed(self):
        try:
            self.manager.predict_localization_model()
            self.refresh_treeview()
        except:
            self.debug(self.localization_dict['prediction_err'])

    def load_localization_dictionary(self):
        """Loads choosen localization file into memory"""
        loc_file_path = os.path.join(self.localization_folder, 'LCGUI_{}.loc'.format(self.localization_mode))
        loc_file = open(loc_file_path, 'r')
        for line in loc_file:
            split = line.split('->')
            self.localization_dict[split[0]] = split[1].split('\n')[0]
        loc_file.close()

    def debug(self, message):
        if (self.debug_output):
            print(message)

    def run(self):
        self.mainwindow.mainloop()

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    root.title("Lung Cancer Detection GUI")
    root.resizable(False, False)
    app = Interface(root)
    app.run()