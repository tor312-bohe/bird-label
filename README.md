# 🦅 Bird Label - AI-Powered Bird Recognition Web App

An intelligent web application that identifies bird species from photos using AI and adds beautiful bilingual captions (Chinese/Latin names) to your images.

## 🌟 Features

- **🤖 AI Bird Recognition** - Powered by HHOLOVE API (懂鸟) for highly accurate species identification
- **🏷️ Automatic Captioning** - Adds beautiful Chinese and Latin name labels to your photos
- **🔍 Search Mode** - Manual search by Chinese name, pinyin, or Latin name
- **🌍 Location Integration** - Uses GPS data for better identification accuracy
- **📱 Responsive Design** - Works seamlessly on desktop and mobile devices
- **🎨 Beautiful UI** - Modern, clean interface with nature-inspired design

## 🚀 Demo

![Bird Label Demo](background.jpg)

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **AI Recognition**: HHOLOVE API integration
- **Database**: SQLite with bird species data
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL (Pillow)
- **Fonts**: Dynamic Chinese font handling

## 📋 Prerequisites

- Python 3.8 or higher
- HHOLOVE API key (get one at [ai.open.hhodata.com](https://ai.open.hhodata.com/))

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/bird-label.git
   cd bird-label
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your HHOLOVE_API_KEY
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:8000
   ```

## 🔧 Configuration

Create a `.env` file with your API configuration:

```env
# HHOLOVE AI API Configuration
HHOLOVE_API_KEY=your_api_key_here

# Flask Configuration
FLASK_ENV=development
DEBUG=True
```

## 📁 Project Structure

```
bird-label/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── background.jpg        # Background image
├── Resources/            # UI icons and assets
│   ├── birdlogo.svg
│   ├── camera.svg
│   └── ...
├── data/                 # Database files
│   └── species.sqlite
├── uploads/              # Uploaded images
├── outputs/              # Generated captioned images
└── fonts/               # Downloaded fonts
```

## 🌐 Deployment

This application can be easily deployed to various platforms:

- **Azure App Service** (recommended)
- **Heroku**
- **Railway**
- **Render**
- **Google Cloud Run**

For production deployment, make sure to:
1. Set `FLASK_ENV=production` in your environment variables
2. Configure your `HHOLOVE_API_KEY`
3. Use a production WSGI server (gunicorn is included)

## 🎯 Usage

1. **Upload a bird photo** by dragging it into the upload area
2. **Wait for automatic location detection** (or enter manually)
3. **Click "Recognize Bird"** to identify the species using AI
4. **Review the identification results** with detailed species information
5. **Click "Label Bird"** to generate a beautifully captioned image
6. **Download or share** your labeled photo

## 🧠 AI Recognition

The app uses the HHOLOVE (懂鸟) API, which provides:
- High accuracy bird species identification
- Comprehensive Chinese bird database
- Detailed encyclopedia information
- Location-based confidence boosting

## 🎨 Customization

You can customize various aspects:
- **Caption styling**: Modify the `add_caption()` function
- **UI themes**: Update the CSS styles
- **Database**: Add more species to the SQLite database
- **Languages**: Extend multilingual support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [HHOLOVE (懂鸟)](https://ai.open.hhodata.com/) for providing the bird recognition API
- Bird photographers who contribute to species databases
- Open source community for the underlying technologies

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Made with ❤️ for bird enthusiasts and nature lovers
# Deployment test Tue Sep 16 21:43:58 CST 2025
