from flask import Flask, render_template, request, jsonify, redirect, flash
import cv2
import numpy as np
from PIL import Image, ImageDraw
import functools, operator
from tensorflow.keras.models import load_model
from werkzeug.utils  import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager , UserMixin , login_required ,login_user, logout_user,current_user

app=Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.db'
app.config['SECRET_KEY']='90123'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin,db.Model):
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    username = db.Column(db.String(200))
    email = db.Column(db.String(200))
    password = db.Column(db.String(500))


@login_manager.user_loader
def get(id):
    return User.query.get(id)

@app.login_manager.unauthorized_handler
def unauth_handler():
    return jsonify(success=False,
                   data={'login_required': True},
                   message='Unauthorized, You need to login first to access this page'), 401

@app.errorhandler(404)
def not_found(error):
  resp = jsonify( { 
    u'status': 404, 
    u'message': u'Resource not found' 
    })
  resp.status_code = 404
  return resp
  
#We can also implement checks and error handling for login and reset.
# ***NOT IMPLEMENTING IT RIGHT NOW*** #
@app.route('/pridict',methods=['GET'])
@login_required
def get_home():
    return render_template('index.html',username=current_user.username)

@app.route('/',methods=['GET'])
def get_login():
    return render_template('login.html')

@app.route('/signup',methods=['GET'])
def get_signup():
    return render_template('signup.html')

@app.route('/reset', methods=['GET'])
def reset():
    # ***NOT IMPLEMENTING IT RIGHT NOW*** #
    return render_template('reset.html')

@app.route('/',methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username,password=password).first()
    if not user:
        flash('Please check your login credentials.')
        return redirect('/')
    login_user(user)
    return redirect('/pridict')

@app.route('/signup',methods=['POST'])
def signup_post():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()   
    if user: # if a user is found, we want to redirect back to signup page so user can try again
        flash('Email address already exists')
        return redirect('/signup')
    new_user = User(email=email, username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    flash('You have signed up successfully')
    return redirect('/signup')

@app.route('/logout',methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect('/')


rgb = [(255, 0, 0), (0, 0, 255), (0, 128, 0), (0, 0, 128), (255, 192, 203), (173, 216, 230), (255, 165, 0), (128, 128, 128), (0, 0, 0), (255, 255, 255)]
def remove_border(im, xwidth, ywidth):
    height, width = im.shape  ## assuming grayscale image
    im[0:ywidth, :] = 0
    im[width-ywidth:width, :] = 0
    im[:, 0:xwidth] = 0
    im[:, height-xwidth:height] = 0
    

def group_by_prop(lst, prop, variation):
    ### Forms group of objects on the basis of prop. Will find the average and check if the new value is in 'variation'
    ### neighbourhood of the current group
    ### prop is a function that can be applied tn 
    sorted_lst = sorted(lst, key = prop)
    groups = [[]]
    sum_group = 0
    num_group = 0
    for elem in sorted_lst:
        num_group = len(groups[-1])
        if num_group == 0:            ### first element is added to first group by default
            groups[-1].append(elem)
            sum_group = prop(elem)
            continue
            
        mean_group = sum_group/num_group
        curr_prop =  prop(elem)
        
        if (abs(mean_group-curr_prop) <= mean_group*variation): # within variation of mean value of prop for current group
            groups[-1].append(elem)
            sum_group += curr_prop
        else:
            #change of group
            groups.append([elem])
            sum_group = curr_prop
            num_group = 1
    return groups

def checkContiguity(candidate):
    curr_x = candidate[0][0]  # x coordinate of first box
    width = candidate[0][2]
    for i in range(1,6):
        next_x = candidate[i][0]
        if abs(curr_x - next_x) >= 1.2*width: #if the boxes are too far, then reject (20% of width)
            return False
        width = candidate[i][2] # new width
        curr_x = next_x
    return True
@app.route('/pridict', methods=['GET', 'POST'])
def pridict():
  f = request.files['file']
  f.save(secure_filename(f.filename))    
  img_file_name = f.filename
  image = cv2.imread(img_file_name)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(3,3),0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  # kernel = np.ones((3,3), np.uint8)
  # dilated = cv2.dilate(thresh,kernel) 
  # print("Image after applying binarizatin")
  # display(Image.fromarray(thresh))
  #print("Image after applying binarizatin")
  cv2.imwrite("static/output/thresh.png", thresh)
  image = cv2.imread(img_file_name)
  cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  rects = []
  for c in cnts:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.015 * peri, True)
      x,y,w,h = cv2.boundingRect(approx)
      if (w*h >= 300):
          rects.append((x,y,w,h))
          cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),1)
  #print("Contours found:")
  cv2.imwrite("static/output/rects.png", image)
# As pin code is usually on the lower right of the post card, we can only consider the rects which are in the lower half.
# right rects. We can achieve this by arranging the rects in increasing order of x and y values, and select those with 
# high values of starting point. 
## ***NOT IMPLEMENTING IT RIGHT NOW*** ##
  # We shall try to find if there are six rectangle of almost equal area and close/contiguous to each other

  rect_area_groups = group_by_prop(rects, lambda r: r[2]*r[3], 0.05)
  y_groups = list()
  for g in rect_area_groups:
      y_groups.extend(group_by_prop(g, lambda r: r[1], 0.1))

  large_y_groups = filter(lambda x: len(x)>= 6, y_groups)
  large_y_groups_x_sorted = list(map(lambda l: sorted(l, key=lambda r: r[0]), large_y_groups)) 

  contiguous_large_y_groups = list()
  for g in large_y_groups_x_sorted:
      if checkContiguity(g):
          contiguous_large_y_groups.append(g)

  im = cv2.imread(img_file_name)
  for index in range(len(contiguous_large_y_groups)):
      candidate = contiguous_large_y_groups[index]
      for rect in candidate:
          x,y,w,h = rect
          cv2.rectangle(im,(x,y),(x+w,y+h), rgb[index], 2)
  cv2.imwrite("static/output/final_rects.png", im)
  #Now we can get the pincode if exists

  model=load_model('model.h5')
  possible_pincodes = []

  for candidate in contiguous_large_y_groups:
      pincode = []
      #print("Candidate --- ")
      for rect_index in range(len(candidate)):
          rect = candidate[rect_index]
          x,y,w,h = rect
          digit_cropped = thresh[y:y+h, x:x+w]
          digit_cropped_resized = cv2.resize(digit_cropped, (28, 28), interpolation=cv2.INTER_AREA)
          kernel = np.ones((3,3), np.uint8)
          digit_cropped_dilated = digit_cropped_resized # cv2.dilate(digit_cropped_resized,kernel) 
          remove_border(digit_cropped_dilated, 2,2)
          final_img = digit_cropped_dilated.astype('float32')/255 
          cv2.imwrite("static/output/pred" + str(rect_index) + ".png", digit_cropped_dilated)
          pred = model.predict(final_img.reshape(1,28, 28, 1))
          pincode.append(pred.argmax())
      possible_pincodes.append(''.join(map(str, pincode)))
  if len(possible_pincodes) == 1:
      return render_template('index.html', pincode=possible_pincodes[0], username=current_user.username)
  else:
      for pin in possible_pincodes:
        return render_template('index.html', pincode=pin, username=current_user.username)
if __name__=='__main__':
    app.run(debug=True)

