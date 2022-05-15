import cv2


def get_face_coordinates(face_coordinates):
    x, y, width, height = face_coordinates
    return (x, x+width, y, y+height)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def resize_image_to_fit_screen(img, orig_shape_x, orig_shape_y):
    if(orig_shape_x > 1000):
        d_factor = orig_shape_x/1000
        orig_shape_x = orig_shape_x//d_factor
        orig_shape_y = orig_shape_y//d_factor
    if(orig_shape_y > 1000):
        d_factor = orig_shape_y/1000
        orig_shape_x = orig_shape_x//d_factor
        orig_shape_y = orig_shape_y//d_factor
    updated_image = cv2.resize(img, (int(orig_shape_y), int(orig_shape_x)))
    return updated_image


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
