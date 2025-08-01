// Main JavaScript file for Personal Website

$(document).ready(function() {
    
    // Smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(event) {
        var target = $(this.getAttribute('href'));
        if (target.length) {
            event.preventDefault();
            $('html, body').stop().animate({
                scrollTop: target.offset().top - 70
            }, 1000);
        }
    });
    
    // Active navigation link highlighting
    function updateActiveNavLink() {
        var currentPage = window.location.pathname.split('/').pop() || 'index.html';
        $('.navbar-nav .nav-link').removeClass('active');
        $('.navbar-nav .nav-link').parent().removeClass('active');
        $('.navbar-nav .nav-link[href="' + currentPage + '"]').addClass('active');
        $('.navbar-nav .nav-link[href="' + currentPage + '"]').parent().addClass('active');
    }
    updateActiveNavLink();
    
    // Contact form handling
    $('#contactForm').on('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        var formData = {
            name: $('#name').val(),
            email: $('#email').val(),
            subject: $('#subject').val(),
            message: $('#message').val()
        };
        
        // Simulate form submission (replace with actual backend integration)
        $('#formMessage').html('<div class="alert alert-info">Sending message...</div>');
        
        setTimeout(function() {
            // Simulate successful submission
            $('#formMessage').html('<div class="alert alert-success">Thank you for your message! I will get back to you soon.</div>');
            $('#contactForm')[0].reset();
            
            // Hide success message after 5 seconds
            setTimeout(function() {
                $('#formMessage').fadeOut();
            }, 5000);
        }, 1500);
    });
    
    // Blog category filtering
    $('.category-filter').on('click', function() {
        var category = $(this).data('category');
        
        // Update active button
        $('.category-filter').removeClass('active');
        $(this).addClass('active');
        
        // Filter blog posts
        if (category === 'all') {
            $('.blog-post').fadeIn();
        } else {
            $('.blog-post').hide();
            $('.blog-post[data-category*="' + category + '"]').fadeIn();
        }
    });
    
    // Add animation to cards on scroll
    function animateCards() {
        $('.card').each(function() {
            var card = $(this);
            var cardTop = card.offset().top;
            var windowBottom = $(window).scrollTop() + $(window).height();
            
            if (cardTop < windowBottom - 100) {
                card.addClass('animated');
            }
        });
    }
    
    // Initial check for cards in view
    animateCards();
    
    // Check on scroll
    $(window).on('scroll', function() {
        animateCards();
    });
    
    // Form validation enhancement
    function validateEmail(email) {
        var re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }
    
    // Real-time email validation
    $('#email').on('blur', function() {
        var email = $(this).val();
        if (email && !validateEmail(email)) {
            $(this).addClass('is-invalid');
            if (!$(this).next('.invalid-feedback').length) {
                $(this).after('<div class="invalid-feedback">Please enter a valid email address.</div>');
            }
        } else {
            $(this).removeClass('is-invalid');
            $(this).next('.invalid-feedback').remove();
        }
    });
    
    // Mobile menu auto-close
    $('.navbar-nav .nav-link').on('click', function() {
        if ($('.navbar-toggler').is(':visible')) {
            $('.navbar-collapse').collapse('hide');
        }
    });
    
    // Add loading state to buttons
    $('button[type="submit"]').on('click', function() {
        var btn = $(this);
        if (btn.closest('form')[0].checkValidity()) {
            btn.prop('disabled', true);
            setTimeout(function() {
                btn.prop('disabled', false);
            }, 3000);
        }
    });
    
    // Enhance Material Icons alignment
    $('.material-icons').addClass('align-middle');
    
    // Theme Toggle Functionality
    function initTheme() {
        // Check for saved theme preference or default to light mode
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeToggle(savedTheme);
    }
    
    function updateThemeToggle(theme) {
        const toggleBtn = $('#themeToggle');
        const icon = toggleBtn.find('i');
        const text = toggleBtn.find('span');
        
        if (theme === 'dark') {
            icon.text('dark_mode');
            text.text('Dark');
            toggleBtn.attr('title', 'Currently in dark mode');
        } else {
            icon.text('light_mode');
            text.text('Light');
            toggleBtn.attr('title', 'Currently in light mode');
        }
    }
    
    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeToggle(newTheme);
        
        // Add smooth transition effect
        $('body').addClass('theme-transitioning');
        setTimeout(() => {
            $('body').removeClass('theme-transitioning');
        }, 300);
    }
    
    // Initialize theme on page load
    initTheme();
    
    // Theme toggle button click handler
    $(document).on('click', '#themeToggle', function(e) {
        e.preventDefault();
        toggleTheme();
    });
    
    // Print resume functionality
    if (window.location.pathname.includes('resume.html')) {
        // Add print button to resume page dynamically
        $('.resume-header .container').append(
            '<div class="text-center mt-3">' +
            '<button class="btn btn-outline-primary" onclick="window.print()">' +
            '<i class="material-icons align-middle">print</i> Print Resume' +
            '</button></div>'
        );
    }
    
    // Back to top button
    var backToTopBtn = $('<button class="btn btn-primary btn-sm" id="backToTop" style="position: fixed; bottom: 20px; right: 20px; display: none; z-index: 1000;">' +
        '<i class="material-icons">arrow_upward</i></button>');
    $('body').append(backToTopBtn);
    
    $(window).scroll(function() {
        if ($(this).scrollTop() > 300) {
            $('#backToTop').fadeIn();
        } else {
            $('#backToTop').fadeOut();
        }
    });
    
    $('#backToTop').click(function() {
        $('html, body').animate({scrollTop: 0}, 800);
        return false;
    });
    
});

// Additional utility functions

// Format date for blog posts
function formatDate(dateString) {
    var options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Character counter for contact form message
$(document).ready(function() {
    $('#message').on('input', function() {
        var length = $(this).val().length;
        var maxLength = 1000;
        
        if (!$('#charCount').length) {
            $(this).after('<small id="charCount" class="form-text text-muted"></small>');
        }
        
        $('#charCount').text(length + ' / ' + maxLength + ' characters');
        
        if (length > maxLength * 0.9) {
            $('#charCount').removeClass('text-muted').addClass('text-warning');
        } else {
            $('#charCount').removeClass('text-warning').addClass('text-muted');
        }
    });
});