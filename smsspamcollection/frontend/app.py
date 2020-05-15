import requests
from flask import Flask, request,render_template

app = Flask(__name__)
app.config['DEBUG'] = True
 


## add models here ....

@app.route('/', methods = ["GET","POST"])
def index():
	if(request.method == "POST"):
		weather_data = []
		url  = "http://api.openweathermap.org/data/2.5/weather?q={}&units=imperial&appid=f185f27d1e1707f6ef57a0db6dae9545"	
		city = request.form['city']
		r = requests.get(url.format(city)).json()
		weather = {
			'city': city,
			'temperature':r['main']['temp'],
			'description':r['weather'][0]['description'],
			'icon' : r['weather'][0]['icon'],
		}
		temperature=r['main']['temp']
		weather_data.append(weather)
		return render_template('result.html',weather_data =weather_data, temperature = temperature)
	else:
		return render_template('mainpage.html')	


@app.route('/aboutme')
def about():
	return render_template('aboutme.html')



if __name__ == "__main__":
	app.run(debug = True)
