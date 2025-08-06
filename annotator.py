import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QMessageBox, QRadioButton, QButtonGroup, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QPoint
import cv2
from PyQt5.QtWidgets import QFontDialog
from PyQt5.QtWidgets import QLineEdit

class YOLOLabelEditor(QWidget):

    COLOR_PALETTE = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Label Editor with Classes")

        self.img_folder = ""
        self.lbl_folder = ""
        self.img_files = []
        self.current_index = -1
        self.boxes = []  # list of (class_id, x_center, y_center, w, h)
        self.img_rgb = None  # Initialize to None
        self.h = 0  # Initialize height
        self.w = 0  # Initialize width

        self.label_classes = [
            "AP_LOGO",
            "BHS_LOGO",
            "Sander",
            "ISafe",
            "Shirt",
            "Spray",
            "SprayMachine"
            # Add all your desired class names here, in the order of their class IDs (0, 1, 2, ...)
        ]
        self.DEFAULT_CLASS = 4 # Default class for initial selection
        self.selected_class_id = self.DEFAULT_CLASS # Will store the ID of the currently selected class

        # UI components
        self.img_label = QLabel("Open image folder to start")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.btn_open_img = QPushButton("Open Image Folder")
        self.btn_open_lbl = QPushButton("Open Label Folder")
        self.btn_load_classes = QPushButton("Load Classes File")
        self.btn_enable_draw = QPushButton("Enable Draw")
        self.btn_prev = QPushButton("Previous Image")
        self.btn_next = QPushButton("Next Image")

        self.btn_enable_draw.setCheckable(True)
        self.btn_enable_draw.setChecked(False)
        self.btn_enable_draw.setStyleSheet("background-color: red;")

        self.lbl_file_label = QLabel("")
        self.image_count_label = QLabel("Image: - / -")
        self.image_count_label.setAlignment(Qt.AlignRight)
        self.image_count_label.setStyleSheet("font-size: 10px; color: gray; padding-right: 4px;")
        self.image_count_label.setFixedHeight(18)

        self.lbl_file_label.setAlignment(Qt.AlignLeft)
        self.lbl_file_label.setStyleSheet("font-size: 10px; color: gray; padding-left: 4px;")
        self.lbl_file_label.setFixedHeight(18)

        self.btn_open_img.clicked.connect(self.open_image_folder)
        self.btn_open_lbl.clicked.connect(self.open_label_folder)
        self.btn_load_classes.clicked.connect(self.load_classes_file)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_enable_draw.clicked.connect(self.toggle_draw_mode)

        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Enter image index or name")
        self.jump_input.setFixedWidth(200)

        self.btn_jump = QPushButton("Go")
        self.btn_jump.clicked.connect(self.jump_to_image)

        # --- Class Selection UI ---
        self.class_radio_layout = QVBoxLayout()
        self.class_button_group = QButtonGroup(self)
        self.class_button_group.buttonToggled.connect(self.on_class_radio_toggled)

        # Use a QWidget to hold the class radio buttons and then put it in a QScrollArea
        self.class_selection_widget = QWidget()
        self.class_selection_widget.setLayout(self.class_radio_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.class_selection_widget)
        self.scroll_area.setFixedWidth(200) # Adjust width as needed

        self.populate_class_radio_buttons() # Initial population

        # --- Layouts ---
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        nav_layout.addWidget(self.jump_input)
        nav_layout.addWidget(self.btn_jump)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_open_img)
        top_layout.addWidget(self.btn_open_lbl)
        top_layout.addWidget(self.btn_load_classes) # Keep this button
        top_layout.addWidget(self.btn_enable_draw)

        # Main layout now includes the class selection on the right
        main_h_layout = QHBoxLayout()
        image_and_nav_layout = QVBoxLayout()
        image_and_nav_layout.addLayout(top_layout)
        image_and_nav_layout.addWidget(self.img_label)
        image_and_nav_layout.insertWidget(2, self.lbl_file_label)
        image_and_nav_layout.insertWidget(3, self.image_count_label)
        image_and_nav_layout.addLayout(nav_layout)

        main_h_layout.addLayout(image_and_nav_layout)
        main_h_layout.addWidget(self.scroll_area) # Add the scroll area with class selection

        self.setLayout(main_h_layout) # Set the main horizontal layout
        self.showMaximized()  # Make the window open maximized


        self.setFocusPolicy(Qt.StrongFocus)

        # Initial assignment of event handlers
        # Always allow mouse move for crosshair, but control visibility
        self.img_label.mouseMoveEvent = self.mouseMoveEvent
        self.img_label.setMouseTracking(True) # Essential for mouseMoveEvent to fire without button press

        self.img_label.mousePressEvent = self.handle_mouse_press # Default to deletion mode
        self.img_label.mouseReleaseEvent = None # No release event for deletion mode initially


        self.drawing_enabled = False
        self.drawing = False
        self.start_point = None
        self.end_point = None

        self.crosshair_pos = QPoint()
        self.show_crosshair = False # Crosshair hidden by default

    def populate_class_radio_buttons(self):
        # Clear existing buttons
        for i in reversed(range(self.class_radio_layout.count())):
            widget = self.class_radio_layout.itemAt(i).widget()
            if widget:
                self.class_radio_layout.removeWidget(widget)
                self.class_button_group.removeButton(widget)
                widget.deleteLater()

        # Add new radio buttons
        for i, class_name in enumerate(self.label_classes):
            radio_button = QRadioButton(f"{i}: {class_name}")
            radio_button.class_id = i # Store the class ID
            self.class_radio_layout.addWidget(radio_button)
            self.class_button_group.addButton(radio_button, i) # Assign ID to the button in the group

            # Set initial selection
            if i == self.selected_class_id:
                radio_button.setChecked(True)

    def on_class_radio_toggled(self, button, checked):
        if checked:
            self.selected_class_id = self.class_button_group.id(button)
            print(f"Selected class ID: {self.selected_class_id} ({self.label_classes[self.selected_class_id]})")


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Delete:  # New condition for Delete key
            self.delete_current_image()
        super().keyPressEvent(event)

    def delete_current_image(self):
        if self.current_index == -1 or not self.img_files:
            QMessageBox.information(self, "No Image", "No image is currently loaded or available to delete.")
            return

        current_img_filename = self.img_files[self.current_index]
        img_path = os.path.join(self.img_folder, current_img_filename)
        lbl_name = os.path.splitext(current_img_filename)[0] + ".txt"
        lbl_path = os.path.join(self.lbl_folder, lbl_name)

        deleted_successfully = False
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                deleted_successfully = True
            else:
                QMessageBox.warning(self, "Image Not Found", f"Image file '{current_img_filename}' not found at path: {img_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Deleting Image", f"Could not delete image file:\n{e}")

        try:
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            else:
                QMessageBox.warning(self, "Label Not Found", f"Label file '{lbl_name}' not found at path: {lbl_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Deleting Label", f"Could not delete label file:\n{e}")

        if deleted_successfully:
            # Remove the deleted file from our list
            self.img_files.pop(self.current_index)

            if not self.img_files:  # No images left
                self.current_index = -1
                self.img_rgb = None
                self.h, self.w = 0, 0
                self.boxes = []
                self.img_label.clear()
                self.lbl_file_label.setText("No images available.")
                self.image_count_label.setText("Image: - / -")
                QMessageBox.information(self, "Done", "All images in the folder have been deleted.")
            else:
                # Adjust current_index if the last image was deleted, otherwise stay at current
                if self.current_index >= len(self.img_files):
                    self.current_index = len(self.img_files) - 1
                self.load_image_and_labels()  # Load the next (or adjusted) image

    def jump_to_image(self):
        value = self.jump_input.text().strip()
        if not value:
            return

        if value.isdigit():
            idx = int(value)
            if 0 <= idx < len(self.img_files):
                self.current_index = idx
                self.load_image_and_labels()
            else:
                QMessageBox.warning(self, "Invalid Index",
                                    f"Index out of range. Valid: 0 to {len(self.img_files)-1}")
            return

        value_lower = value.lower()
        for idx, fname in enumerate(self.img_files):
            base = os.path.splitext(fname)[0].lower()
            if base == value_lower or fname.lower() == value_lower:
                self.current_index = idx
                self.load_image_and_labels()
                return

        QMessageBox.warning(self, "Not Found", f"No image named '{value}' found in folder.")

    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.img_folder = folder
            self.img_files = sorted([f for f in os.listdir(folder)
                                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            if not self.img_files:
                QMessageBox.warning(self, "Warning", "No image files found in folder")
                return
            self.current_index = 0
            self.load_image_and_labels()

    def open_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Label Folder")
        if folder:
            self.lbl_folder = folder
            if self.current_index >= 0:
                self.load_image_and_labels()

    def load_classes_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Classes File", filter="Text Files (*.txt)")
        if path:
            try:
                with open(path, 'r') as f:
                    self.label_classes = [line.strip() for line in f if line.strip()]
                QMessageBox.information(self, "Classes Loaded",
                                        f"Loaded {len(self.label_classes)} classes.")
                self.populate_class_radio_buttons() # Re-populate radio buttons
                self.update_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load classes file:\n{e}")

    def load_image_and_labels(self):
        if self.current_index < 0 or self.current_index >= len(self.img_files):
            self.img_rgb = None  # Clear image if index is invalid
            self.h, self.w = 0, 0
            self.boxes = []
            self.update_display()
            return

        img_path = os.path.join(self.img_folder, self.img_files[self.current_index])
        self.img = cv2.imread(img_path)
        if self.img is None:
            QMessageBox.warning(self, "Warning", f"Cannot load image: {img_path}")
            self.img_rgb = None  # Clear image if loading fails
            self.h, self.w = 0, 0
            return
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]

        self.boxes = []
        if self.lbl_folder:
            lbl_name = os.path.splitext(self.img_files[self.current_index])[0] + ".txt"
            lbl_path = os.path.join(self.lbl_folder, lbl_name)
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_c, y_c, bw, bh = map(float, parts[1:])
                            self.boxes.append([class_id, x_c, y_c, bw, bh])

        self.update_display()

        # Reset drawing related flags when loading a new image
        self.btn_enable_draw.setChecked(False)
        self.drawing_enabled = False
        self.btn_enable_draw.setStyleSheet("background-color: red;")
        # Set event handlers back to deletion mode
        self.img_label.mousePressEvent = self.handle_mouse_press
        self.img_label.mouseReleaseEvent = None # No release event for deletion mode
        # self.img_label.mouseMoveEvent is handled globally now and its effect depends on drawing_enabled
        self.img_label.setCursor(Qt.ArrowCursor)  # Reset cursor
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.show_crosshair = False  # Reset crosshair state

        if self.lbl_folder:
            lbl_name = os.path.splitext(self.img_files[self.current_index])[0] + ".txt"
            lbl_path = os.path.join(self.lbl_folder, lbl_name)
            if os.path.exists(lbl_path):
                self.lbl_file_label.setText(f"Label file: {lbl_name}")
            else:
                self.lbl_file_label.setText(f"Label file: {lbl_name} (Not found)")
        else:
            self.lbl_file_label.setText("Label file: (Label folder not selected)")
        total = len(self.img_files)
        current = self.current_index + 1 if self.current_index >= 0 else "-"
        self.image_count_label.setText(f"Image: {current} / {total}")

    def update_display(self):
        if self.img_rgb is None:
            self.img_label.clear()
            return

        disp_img = self.img_rgb.copy()

        for i, (cls, x_c, y_c, bw, bh) in enumerate(self.boxes):
            x1 = int((x_c - bw / 2) * self.w)
            y1 = int((y_c - bh / 2) * self.h)
            x2 = int((x_c + bw / 2) * self.w)
            y2 = int((y_c + bh / 2) * self.h)

            color = self.get_color_for_class(cls)
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), color, 2)
            label_text = self.label_classes[cls] if 0 <= cls < len(self.label_classes) else str(cls)
            cv2.putText(disp_img, label_text,
                        (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Draw temporary rectangle while dragging ---
        if self.drawing and self.start_point and self.end_point:
            label_size = self.img_label.size()
            image_aspect_ratio = self.w / self.h

            if (label_size.width() / label_size.height()) > image_aspect_ratio:
                displayed_h = label_size.height()
                displayed_w = int(displayed_h * image_aspect_ratio)
            else:
                displayed_w = label_size.width()
                displayed_h = int(displayed_w / image_aspect_ratio)

            x_offset = (label_size.width() - displayed_w) // 2
            y_offset = (label_size.height() - displayed_h) // 2

            if displayed_w == 0 or displayed_h == 0:  # Avoid ZeroDivisionError
                # This should ideally not happen if img_rgb is valid
                return

            scale_factor_x = self.w / displayed_w
            scale_factor_y = self.h / displayed_h

            start_x_relative_to_displayed = self.start_point.x() - x_offset
            start_y_relative_to_displayed = self.start_point.y() - y_offset
            end_x_relative_to_displayed = self.end_point.x() - x_offset
            end_y_relative_to_displayed = self.end_point.y() - y_offset

            temp_x1_orig = int(max(0, min(start_x_relative_to_displayed * scale_factor_x, end_x_relative_to_displayed * scale_factor_x)))
            temp_y1_orig = int(max(0, min(start_y_relative_to_displayed * scale_factor_y, end_y_relative_to_displayed * scale_factor_y)))
            temp_x2_orig = int(min(self.w, max(start_x_relative_to_displayed * scale_factor_x, end_x_relative_to_displayed * scale_factor_x)))
            temp_y2_orig = int(min(self.h, max(start_y_relative_to_displayed * scale_factor_y, end_y_relative_to_displayed * scale_factor_y)))

            cv2.rectangle(disp_img, (temp_x1_orig, temp_y1_orig), (temp_x2_orig, temp_y2_orig), (0, 0, 255), 2)

        # --- Draw Crosshair ---
        # The crosshair should only be drawn if drawing_enabled is True AND the mouse is over the image area
        if self.show_crosshair and self.crosshair_pos and self.w > 0 and self.h > 0:
            label_size = self.img_label.size()
            image_aspect_ratio = self.w / self.h

            if (label_size.width() / label_size.height()) > image_aspect_ratio:
                displayed_h = label_size.height()
                displayed_w = int(displayed_h * image_aspect_ratio)
            else:
                displayed_w = label_size.width()
                displayed_h = int(displayed_w / image_aspect_ratio)

            x_offset = (label_size.width() - displayed_w) // 2
            y_offset = (label_size.height() - displayed_h) // 2

            if displayed_w == 0 or displayed_h == 0:  # Avoid ZeroDivisionError
                return

            scale_factor_x = self.w / displayed_w
            scale_factor_y = self.h / displayed_h

            mouse_x_rel_displayed = self.crosshair_pos.x() - x_offset
            mouse_y_rel_displayed = self.crosshair_pos.y() - y_offset

            crosshair_x_orig = int(mouse_x_rel_displayed * scale_factor_x)
            crosshair_y_orig = int(mouse_y_rel_displayed * scale_factor_y)

            crosshair_x_orig = max(0, min(crosshair_x_orig, self.w - 1))
            crosshair_y_orig = max(0, min(crosshair_y_orig, self.h - 1))

            cv2.line(disp_img, (crosshair_x_orig, 0), (crosshair_x_orig, self.h - 1), (0, 255, 255), 1)  # Cyan
            cv2.line(disp_img, (0, crosshair_y_orig), (self.w - 1, crosshair_y_orig), (0, 255, 255), 1)  # Cyan

        h_disp, w_disp, ch_disp = disp_img.shape
        bytes_per_line = ch_disp * w_disp
        qt_img = QImage(disp_img.data, w_disp, h_disp, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio)
        self.img_label.setPixmap(pix)

    def get_color_for_class(self, class_id):
        return self.COLOR_PALETTE[class_id % len(self.COLOR_PALETTE)]

    def handle_mouse_press(self, event):
        # This function is ONLY for deleting boxes (when drawing is DISABLED)
        if self.drawing_enabled:
            return

        # Check if an image is loaded, if not, return
        if self.img_rgb is None or self.w == 0 or self.h == 0:
            return

        pixmap = self.img_label.pixmap()
        if pixmap is None:
            return

        label_size = self.img_label.size()

        image_aspect_ratio = self.w / self.h

        if (label_size.width() / label_size.height()) > image_aspect_ratio:
            displayed_h = label_size.height()
            displayed_w = int(displayed_h * image_aspect_ratio)
        else:
            displayed_w = label_size.width()
            displayed_h = int(displayed_w / image_aspect_ratio)

        x_offset = (label_size.width() - displayed_w) // 2
        y_offset = (label_size.height() - displayed_h) // 2

        if displayed_w == 0 or displayed_h == 0:  # Avoid ZeroDivisionError
            return

        scale_factor_x = self.w / displayed_w
        scale_factor_y = self.h / displayed_h

        x_relative_to_displayed = event.x() - x_offset
        y_relative_to_displayed = event.y() - y_offset

        x_orig = x_relative_to_displayed * scale_factor_x
        y_orig = y_relative_to_displayed * scale_factor_y

        x_orig = max(0.0, min(x_orig, self.w))
        y_orig = max(0.0, min(y_orig, self.h))

        for i, (cls, x_c, y_c, bw, bh) in enumerate(self.boxes):
            x1 = (x_c - bw / 2) * self.w
            y1 = (y_c - bh / 2) * self.h
            x2 = (x_c + bw / 2) * self.w
            y2 = (y_c + bh / 2) * self.h
            if x1 <= x_orig <= x2 and y1 <= y_orig <= y2:
                label_text = self.label_classes[cls] if 0 <= cls < len(self.label_classes) else str(cls)
                reply = QMessageBox.question(self, "Delete Box",
                                             f"Delete bounding box for class '{label_text}'?",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.boxes.pop(i)
                    self.save_labels()
                    self.update_display()
                break

    def save_labels(self):
        if not self.lbl_folder or self.current_index < 0:
            return
        lbl_name = os.path.splitext(self.img_files[self.current_index])[0] + ".txt"
        lbl_path = os.path.join(self.lbl_folder, lbl_name)
        with open(lbl_path, 'w') as f:
            for cls, x_c, y_c, bw, bh in self.boxes:
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    def next_image(self):
        if self.current_index + 1 < len(self.img_files):
            self.current_index += 1
            self.load_image_and_labels()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image_and_labels()

    def toggle_draw_mode(self):
        self.drawing_enabled = self.btn_enable_draw.isChecked()
        print('Toggle mode', self.drawing_enabled) # Debug print
        if self.drawing_enabled:
            self.btn_enable_draw.setStyleSheet("background-color: green;")
            # Assign drawing-related event handlers
            self.img_label.mousePressEvent = self.handle_draw_press
            self.img_label.mouseReleaseEvent = self.handle_draw_release
            self.img_label.setCursor(Qt.CrossCursor)
            self.show_crosshair = True # Show crosshair when drawing is enabled
            self.update_display() # Update to show crosshair immediately
        else:
            self.btn_enable_draw.setStyleSheet("background-color: red;")
            # Assign deletion-related event handlers
            self.img_label.mousePressEvent = self.handle_mouse_press
            self.img_label.mouseReleaseEvent = None # No release event for deletion mode
            self.img_label.setCursor(Qt.ArrowCursor)
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.show_crosshair = False  # Hide crosshair when mode is disabled
            self.update_display()

    def handle_draw_press(self, event):
        if event.button() == Qt.LeftButton:
            # Ensure an image is loaded before attempting to draw
            if self.img_rgb is None or self.w == 0 or self.h == 0:
                QMessageBox.warning(self, "No Image Loaded", "Please load an image before drawing.")
                return

            self.drawing = True
            self.start_point = event.pos()
            self.end_point = None # Reset end point at the start of a new draw
            self.update_display()

    def handle_draw_release(self, event):
        print('Mouse release event', self.drawing) # Debug print
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()

            # IMPORTANT: Reset drawing flag immediately.
            # This ensures that if a QMessageBox pops up,
            # the temporary box is no longer drawn.
            self.drawing = False

            if self.start_point and self.end_point:
                if self.img_label.pixmap() is None or self.img_rgb is None or self.w == 0 or self.h == 0:
                    self.start_point = None
                    self.end_point = None
                    self.update_display()
                    return

                label_size = self.img_label.size()
                image_aspect_ratio = self.w / self.h

                if (label_size.width() / label_size.height()) > image_aspect_ratio:
                    displayed_h = label_size.height()
                    displayed_w = int(displayed_h * image_aspect_ratio)
                else:
                    displayed_w = label_size.width()
                    displayed_h = int(displayed_w / image_aspect_ratio)

                x_offset = (label_size.width() - displayed_w) // 2
                y_offset = (label_size.height() - displayed_h) // 2

                if displayed_w == 0 or displayed_h == 0:
                    self.start_point = None
                    self.end_point = None
                    self.update_display()
                    return

                scale_factor_x = self.w / displayed_w
                scale_factor_y = self.h / displayed_h

                start_x_relative_to_displayed = self.start_point.x() - x_offset
                start_y_relative_to_displayed = self.start_point.y() - y_offset
                end_x_relative_to_displayed = self.end_point.x() - x_offset
                end_y_relative_to_displayed = self.end_point.y() - y_offset

                x1_orig = start_x_relative_to_displayed * scale_factor_x
                y1_orig = start_y_relative_to_displayed * scale_factor_y
                x2_orig = end_x_relative_to_displayed * scale_factor_x
                y2_orig = end_y_relative_to_displayed * scale_factor_y

                x1_final = int(min(x1_orig, x2_orig))
                y1_final = int(min(y1_orig, y2_orig))
                x2_final = int(max(x1_orig, x2_orig))
                y2_final = int(max(y1_orig, y2_orig))

                x1_final = max(0, min(x1_final, self.w))
                y1_final = max(0, min(y1_final, self.h))
                x2_final = max(0, min(x2_final, self.w))
                y2_final = max(0, min(y2_final, self.h))

                box_width_orig = x2_final - x1_final
                box_height_orig = y2_final - y1_final
                center_x_orig = (x1_final + x2_final) / 2.0
                center_y_orig = (y1_final + y2_final) / 2.0

                if box_width_orig <= 5 or box_height_orig <= 5: # Minimum box size
                    self.start_point = None
                    self.end_point = None
                    self.update_display() # Update to remove the temporary box
                    print('Box size is too small')
                    return

                yolo_xc = center_x_orig / self.w
                yolo_yc = center_y_orig / self.h
                yolo_bw = box_width_orig / self.w
                yolo_bh = box_height_orig / self.h

                reply = QMessageBox.question(self, "Add Box",
                                             f"Add this bounding box with class '{self.label_classes[self.selected_class_id] if 0 <= self.selected_class_id < len(self.label_classes) else self.selected_class_id}'?",
                                             QMessageBox.Yes | QMessageBox.No)

                if reply == QMessageBox.Yes:
                    class_to_assign = self.selected_class_id
                    if not self.label_classes or class_to_assign >= len(self.label_classes):
                        QMessageBox.warning(self, "Class ID Issue", f"Selected class ID {class_to_assign} is out of bounds for current classes. Assigning ID 0.")
                        class_to_assign = 0

                    self.boxes.append([class_to_assign, yolo_xc, yolo_yc, yolo_bw, yolo_bh])
                    self.save_labels()

                self.start_point = None
                self.end_point = None
                self.update_display() # Final update to show the permanent box or clear temporary if rejected.

    def mouseMoveEvent(self, event):
        # Update crosshair position regardless of drawing_enabled, but visibility is controlled by self.show_crosshair
        # This event is a method of the QWidget (YOLOLabelEditor), but it's passed through via img_label.mouseMoveEvent
        # So event.pos() is relative to the img_label here.

        if self.img_rgb is not None and self.w > 0 and self.h > 0:
            pixmap = self.img_label.pixmap()
            if pixmap and not pixmap.isNull():
                label_rect = self.img_label.contentsRect()

                # Calculate the actual displayed image dimensions and position within the QLabel
                image_aspect_ratio = self.w / self.h
                if (label_rect.width() / label_rect.height()) > image_aspect_ratio:
                    displayed_h = label_rect.height()
                    displayed_w = int(displayed_h * image_aspect_ratio)
                else:
                    displayed_w = label_rect.width()
                    displayed_h = int(displayed_w / image_aspect_ratio)

                x_offset = (label_rect.width() - displayed_w) // 2
                y_offset = (label_rect.height() - displayed_h) // 2

                displayed_image_rect = QRect(x_offset, y_offset, displayed_w, displayed_h)

                # Only update crosshair position and show it if mouse is within the displayed image area
                if displayed_image_rect.contains(event.pos()):
                    self.crosshair_pos = event.pos()
                    # Only show crosshair if drawing mode is enabled
                    self.show_crosshair = self.drawing_enabled
                else:
                    self.show_crosshair = False
            else:
                self.show_crosshair = False
        else:
            self.show_crosshair = False # No image, no crosshair

        # This part handles the temporary drawing rectangle and must always respond to mouse movement
        if self.drawing and self.start_point:
            self.end_point = event.pos()

        # Always update display to reflect crosshair or temporary box changes
        self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOLabelEditor()
    window.show()
    sys.exit(app.exec_())