# Personal Resume & Blog Website

A professional portfolio website featuring resume, blog, and contact sections built with Bootstrap 4.5.2 and Material Icons.

## Features

- **Responsive Design**: Mobile-first approach using Bootstrap
- **Blog System**: Individual blog posts with categorization
- **Material Icons**: Google Material Design icons for visual enhancement
- **Clean UI**: Professional layout with modern styling

## Tech Stack

- HTML5
- CSS3 (Custom styling)
- Bootstrap 4.5.2 (CDN)
- Material Icons (CDN)
- JavaScript/jQuery
- No build process required - pure static site

## Project Structure

```
Me_Resume/
├── index.html              # Home page
├── resume.html             # Resume/CV page
├── blog.html               # Blog listing page
├── contact.html            # Contact form page
├── css/
│   └── style.css          # Custom styles
├── js/
│   └── main.js            # Custom JavaScript
├── images/                 # Image assets
└── blog-post/             # Individual blog posts
    ├── ai-supply-chain-2024.html
    ├── aws-migration.html
    ├── security-zero-trust.html
    └── ... (more blog posts)
```

## How to Run

Since this is a static website with CDN dependencies, you have several options:

### Option 1: Direct File Opening (Simplest)
1. Navigate to the project folder
2. Double-click on `index.html` to open in your default browser

### Option 2: Using Python HTTP Server (Recommended)

**Easy Way (Windows):**
- Double-click `start-server.bat` in the project folder
- Or run `start-server.ps1` in PowerShell

**Manual Way:**
```bash
# Make sure you're in the project directory first
cd "C:\Users\User\Documents\Websites\Me_Resume"

# Python 3 (most common)
python -m http.server 8000

# If python doesn't work, try:
python3 -m http.server 8000

# Or if you have Python 2:
python -m SimpleHTTPServer 8000
```
Then navigate to `http://localhost:8000`

**Troubleshooting:**
- If you get "python is not recognized", Python might not be installed or not in your PATH
- Try opening Command Prompt as Administrator
- Make sure you're in the correct directory before running the command

### Option 3: Using Node.js HTTP Server
```bash
# Install http-server globally (one time)
npm install -g http-server

# Run server
http-server -p 8000
```
Then navigate to `http://localhost:8000`

### Option 4: Using VS Code Live Server Extension
1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

## Dependencies

All dependencies are loaded via CDN, so no installation is required:

- **Bootstrap 4.5.2**: `https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css`
- **Material Icons**: `https://fonts.googleapis.com/icon?family=Material+Icons`
- **jQuery 3.5.1**: `https://code.jquery.com/jquery-3.5.1.slim.min.js`
- **Bootstrap JS**: `https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js`

## Customization

### Updating Personal Information
1. Edit `index.html` to update the hero section with your name and title
2. Update `resume.html` with your experience and qualifications
3. Modify `contact.html` with your contact information

### Adding New Blog Posts
1. Create a new HTML file in the `blog-post/` directory
2. Use existing blog posts as templates
3. Add a link to your new post in `blog.html`

### Styling
- Custom styles are in `css/style.css`
- Override Bootstrap classes or add new styles as needed

## Blog Navigation

- Main blog page: `/blog.html` - Lists all blog posts with categories
- Individual posts: `/blog-post/[post-name].html`
- Each blog post has navigation back to the main site and blog listing

## Browser Compatibility

The site is compatible with all modern browsers:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Internet Explorer 11 (limited support)

## Future Enhancements

- [ ] Add a build process for optimization
- [ ] Implement a contact form backend
- [ ] Add blog search functionality
- [ ] Implement blog post pagination
- [ ] Add dark mode toggle
- [ ] Integrate with a CMS for easier blog management

## License

This project is available for personal use. Please update with your own content and information.

## Author

[Your Name] - Update this with your information

---

**Note**: Remember to update all placeholder content (like "Your Name") with your actual information before deploying.
