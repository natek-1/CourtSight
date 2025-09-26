

import cv2


from utils.bbox_utils import get_bbox_center, get_bbox_width



def draw_ellipse(frame, bbox, color, thickness=2, track_id=None, ):
    _,_,_,y2=bbox
    y2 = int(y2)
    x_center, _ = get_bbox_center(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(frame,
                center=(x_center, y2),
                axes=(int(width), int(0.35*width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235, color =color, thickness=thickness,
                lineType=cv2.LINE_4)
    
    if track_id is not None:
        rectangle_width = 40
        rectangle_heigh = 20
        x1_rect = x_center-rectangle_width//2
        x2_rect = x_center+rectangle_width//2
        y1_rect = y2-rectangle_heigh//2 +15
        y2_rect = y2+rectangle_heigh//2 + 15
        
        cv2.rectangle(
            frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)),
            color, cv2.FILLED
        )
        
        x1_text = x1_rect+12
        if track_id > 99:
            x1_text -=10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )
        
    return frame
    