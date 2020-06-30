from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://root:''@localhost/db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))
    title = db.Column(db.String(20))
    description = db.Column(db.String(50))

    def create(self):
        db.session.add(self)
        db.session.commit()
        return self

    def __init__(self, name, title, description):
        self.name = name
        self.title = title
        self.description = description

    def __repr__(self):
        return '<Post %d>' % self.name


db.create_all()


class PostSchema(ma.Schema):
    class Meta:
        fields = ("name", "title", "description")


post_schema = PostSchema()
posts_schema = PostSchema(many=True)


@app.route('/post', methods=['POST'])
def add_post():
    name = request.json['name']
    title = request.json['title']
    description = request.json['description']

    main_post = Post(name, title, description)
    db.session.add(main_post)
    db.session.commit()
    return post_schema.jsonify(main_post)


@app.route('/post', methods=['GET'])
def index():
    get_names = Post.query.all()
    posts, error = posts_schema.dump(get_names)
    return make_response(jsonify({"Post": posts}))


if __name__ == "__main__":
    app.run(debug=True)
