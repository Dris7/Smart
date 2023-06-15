# Smart Server-Side API

This API is built using Flask framework and is responsible for handling the server-side processing of images for computer vision and image processing tasks. It provides two main routes: `/upload` and `/calculate`.

## Installation

1. Clone the repository:

```shell
git clone <repository_url>
```
2. Install the required dependencies:

```shell
pip install -r requirements.txt
```
3. Run the server:

```shell
python app.py
```
4. Run the server:
The server will now be running on ```http://localhost:5000.```

## Notes
- Ensure that the uploaded image is in a supported format (e.g., JPEG, PNG) and contains a clear representation of the equation.
- The image processing and equation recognition algorithms are implemented using OpenCV and other computer vision techniques. Adjustments or improvements   to these algorithms can be made in the corresponding code files.

## Routes

` /upload - POST `
This route is used to upload an image to the server for processing. It expects a file named `file` in the request form data. Upon successful upload, it will return the equation as a JSON response.

` /calculate - GET`
This route is used to calculate the solution for a given equation. It expects the equation as a query parameter named `equation`. It will return the solution as a JSON response.

## License
This project is licensed under the MIT License.

