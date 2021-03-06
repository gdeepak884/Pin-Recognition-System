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


def checkContiguity(candidate):
    curr_x = candidate[0][0]  # x coordinate of first box
    width = candidate[0][2]
    for i in range(1,6):
        next_x = candidate[i][0]
        if abs(curr_x - next_x) >= 1.2*width: #if the boxes are too far, then reject
            return False
        width = candidate[i][2] # new width
        curr_x = next_x
    return True

@app.route('/pridict', methods=['GET', 'POST'])
@login_required
def pridict():
  f = request.files['file']
  f.save(secure_filename(f.filename))    
  img_file_name = f.filename
  image = cv2.imread(img_file_name)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(3,3),0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  #print("Image after applying binarizatin")
  #display(Image.fromarray(thresh))
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
  #display(Image.fromarray(image)) 
# As pin code is usually on the lower right of the post card, we can only consider the rects which are in the lower half.
# right rects. We can achieve this by arranging the rects in increasing order of x and y values, and select those with 
# high values of starting point. 
## ***NOT IMPLEMENTING IT RIGHT NOW*** ##
  # We shall try to find if there are six rectangle of almost equal area and close/contiguous to each other

  sorted_rects = sorted(rects, key=lambda r: (r[1], r[0]))

# First let us group the boxes based on equal value of almost equal value of Y coordinate (r[1]) 
  y_groups = [[]]
  sum_gy = 0
  num_gy = 0
  for rec in sorted_rects:
      num_gy = len(y_groups[-1])
      if num_gy == 0:
          y_groups[-1].append(rec)
          sum_gy = rec[1]
          continue
      mean_gy = sum_gy/num_gy
      curr_y = rec[1]
      if (abs(mean_gy-curr_y) <= mean_gy*0.1): # within 10% of mean value of y for current group
          y_groups[-1].append(rec)
          sum_gy += curr_y
      else:
        #change of group
          y_groups.append([rec])
          sum_gy = rec[1]
          num_gy = 1
##print(y_groups)
  # we have grouped by Y coordinate, now we will find those groups which are larger than size of six and contain contiguous six boxes
  large_y_groups = filter(lambda x: len(x)>= 6, y_groups)
## sorted_large_y_groups = map(lambda x: sorted(x, key = lambda r: r[0]), large_y_groups)

# among these large groups, let us only keep those which are of almost equal area
  area_sorted_groups = list(map(lambda ls: sorted(ls, key = lambda r: r[2]*r[3]), large_y_groups))  # area wise sorting
# We should group all the areas like we did for y coordinate
  area_groups = [[]]     
  for group in area_sorted_groups:
        sum_gy = 0
        num_gy = 0
        for rec in group:
            num_gy = len(area_groups[-1])
            if num_gy == 0:
                area_groups[-1].append(rec)
                sum_gy = rec[2]*rec[3]
                continue
            mean_gy = sum_gy/num_gy
            curr_y = rec[2]*rec[3]
            if (abs(mean_gy-curr_y) <= mean_gy*0.2): # within 10% of mean value of area for current group
                area_groups[-1].append(rec)
                sum_gy += curr_y
            else:
            #change of group
              area_groups.append([rec])
              sum_gy = rec[2]*rec[3]
              num_gy = 1
  #print(area_groups)
  #for i in area_groups:
        #print(len(i))
# We will again filter out the groups of size larger than 6
  final_rectangle_groups = list(filter(lambda x: len(x)>= 6, area_groups))  # might contain pin 
  #print(final_rectangle_groups)
# Now we need to check for contiguity of the boxes...
# We need to take all the window of six in each group and then check for consecutiveness...
  final_pin_candidates = []
  for group in final_rectangle_groups:
        group = sorted(group, key = lambda rect: rect[0]) # sorting by x-value    
        for pos in range(len(group) - 5):
            candidate = group[pos:pos+6]
            if checkContiguity(candidate):
                final_pin_candidates.append(candidate)
  image = cv2.imread(img_file_name)
  for candidate in final_pin_candidates:
        for rect in candidate:
            x,y,w,h = rect
            cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12),1)
  #display(Image.fromarray(image))
 #Now we can get the pincode if exists
  model=load_model('model.h5')
  possible_pincodes = [] 
  for candidate in final_pin_candidates:
      pincode = []
      for rect in candidate:
          x,y,w,h = rect
          digit_cropped = thresh[y:y+h, x:x+w]
          digit_cropped_resized = cv2.resize(digit_cropped, (28, 28), interpolation=cv2.INTER_AREA)
          digit_cropped_dilated = cv2.dilate(digit_cropped_resized,(3,3))
          final_img = digit_cropped_dilated.astype('float32')/255 
          #display(Image.fromarray(digit_cropped_dilated))
          pred = model.predict(final_img.reshape(1,28, 28, 1))
          pincode.append(pred.argmax())
      possible_pincodes.append(''.join(map(str, pincode)))
  if len(possible_pincodes) == 1:
      resp = jsonify({
      u'status': 200,
      u'pin': possible_pincodes[0]   
      })
      resp.status_code = 200
      return resp
  else:
      for pin in possible_pincodes:
        resp = jsonify({
        u'status': 200,
        u'pin': pin   
        })
        resp.status_code = 200
        return resp    

if __name__=='__main__':
    app.run(debug=True)
