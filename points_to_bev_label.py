def points_to_bev_label(file, label_file, list_roi_xyz=[0.0, 69.12, -21.6, 21,6, -2.5, 0.5], list_voxel_xy=[0.48, 0.30], list_grid_xy = [144, 144]):
    if isinstance(file, str):
        point_dict, _ = get_lane_points(file)
    if isinstance(file, dict):
        point_dict = file
    if point_dict is None:
        print('Invaild file:', file)
        return
    x_min, x_max, y_min, y_max, z_min, z_max = list_roi_xyz
    x_voxel, y_voxel = list_voxel_xy
    label_img = np.full(list_grid_xy, 255, dtype=np.uint8)

    for k, v in point_dict.items():
        if len(v) == 0:
            break
        points = np.array(v)
        idx = np.where(points[:, 0], points[:, 0] >= x_min, points[:, 0] <= x_max)

        points = points[np.where(points[:, 0], points[:, 0] >= x_min, points[:, 0] <= x_max)]
        points = points[np.where(points[:, 1], points[:, 1] >= y_min, points[:, 1] <= y_max)]

        if point.shape[0] == 0:
            break
        x_img = ((points[:, 0] - x_min) // x_voxel).astype(int)
        y_img = ((points[:, 1] - y_min) // y_voxel).astype(int)
        for i, x in enumerate(x_img):
            if x_img[i] < 144 and y_img[i] < 144:
                label_img[x_img[i], y_img[i]] = k
            else:
                print(file)
                print(x_img[i], y_img[i])
                print(points[i, 0], points[i, 1])
        
        if type(k) == str:
            k = eval(k)
        for i in range(len(x_img)-1):
            cv.line(label_img, (y_img[i], x_img[i]), (y_img[i+1], x_img[i+1]), k, 1)

        txt = '{}'.format(k)
        cv2.putText(label_img, txt, (y_img[0], x_img[0] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (10, 255, 255), 1)

    filp_img = np.filp(np.filp(label_img, 0), 1).copy()
    return filp_img

def beving_from_points(self, lane_data):
    laneline_type = [1, 4]
    road_boundary_type = [3, 4]
    bev_range = [69.12, 0.0, 20.16, -20.16]
    grid_size = [0.48, 0.28]
    x_max, x_min, y_max, y_min = bev_range
    grid_x, grid_y = grid_size
    bev_img_h, bev_img_w = int((x_max- x_min) / grid_x), int((y_max - y_min) / gird_y)
    bev_img = np.full((bev_img_h, bev_img_w), 255, dtype=np.uint8)

    for lm in lane_data['lanemarking']:
        pts = np.array(lm['points'])
        for i in range(len(pts) - 1):
            p1_x = int((-pts[i][1] - y_min) // grid_y)
            p1_y = int((-pts[i][0] - x_max) // grid_x)
            p2_x = int((-pts[i+1][1] - y_min) // grid_y)
            p2_y = int((-pts[i+1][0] - x_max) // grid_x)
            if pts[i+1][3] in laneline_type:
                cv2.line(bev_img, (p1_x, p1_y), (p2_x, p2_y), 0, 1)
            if pts[i+1][3] in road_boundary_type:
                cv2.line(bev_img,  (p1_x, p1_y), (p2_x, p2_y), 1, 1)

    return bev_img

