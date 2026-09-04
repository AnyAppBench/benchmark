#!/bin/bash

# ===========================================
# Android World App Setup Script
# Grants permissions + dismisses first-run dialogs
# ===========================================

echo "========================================"
echo "Android World - Complete App Setup"
echo "========================================"
echo ""

# Get emulator device. Override for shared hosts, for example:
# ADB="adb -P 5038 -s emulator-5558" ./benchmark/setup_apps.sh
ADB="${ADB:-adb}"

# ===========================================
# PART 1: Grant Runtime Permissions
# ===========================================

grant_permissions() {
    local pkg=$1
    local name=$2
    
    echo "Granting permissions for $name ($pkg)..."
    
    # Common permissions
    $ADB shell pm grant "$pkg" android.permission.READ_EXTERNAL_STORAGE 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.WRITE_EXTERNAL_STORAGE 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.CAMERA 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.RECORD_AUDIO 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_CONTACTS 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.WRITE_CONTACTS 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_CALENDAR 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.WRITE_CALENDAR 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_SMS 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.SEND_SMS 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_PHONE_STATE 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.CALL_PHONE 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.ACCESS_FINE_LOCATION 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.ACCESS_COARSE_LOCATION 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.ACCESS_BACKGROUND_LOCATION 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.POST_NOTIFICATIONS 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_MEDIA_IMAGES 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_MEDIA_VIDEO 2>/dev/null || true
    $ADB shell pm grant "$pkg" android.permission.READ_MEDIA_AUDIO 2>/dev/null || true
    
    echo "  ✓ Done"
}

echo "--- Granting permissions ---"
grant_permissions "com.android.chrome" "Chrome"
grant_permissions "com.google.android.googlequicksearchbox" "Google"
grant_permissions "net.gsantner.markor" "Markor"
grant_permissions "com.arduia.expense" "Pro Expense"
grant_permissions "org.videolan.vlc" "VLC"
grant_permissions "code.name.monkey.retromusic" "Retro Music"
grant_permissions "com.simplemobiletools.gallery.pro" "Simple Gallery"
grant_permissions "com.simplemobiletools.draw.pro" "Simple Draw"
grant_permissions "com.simplemobiletools.calendar.pro" "Simple Calendar"
grant_permissions "com.simplemobiletools.smsmessenger" "Simple SMS"
grant_permissions "com.dimowner.audiorecorder" "Audio Recorder"
grant_permissions "net.osmand.plus" "OsmAnd"
grant_permissions "com.flauschcode.broccoli" "Broccoli"
grant_permissions "net.cozic.joplin" "Joplin"
grant_permissions "org.tasks" "Tasks"
grant_permissions "de.dennisguse.opentracks" "OpenTracks"
grant_permissions "com.google.android.apps.accessibility.voiceaccess" "AndroidWorld A11y"
grant_permissions "com.google.android.contacts" "Contacts"
grant_permissions "com.android.contacts" "Contacts (AOSP)"
grant_permissions "com.google.android.dialer" "Phone"
grant_permissions "com.android.dialer" "Phone (AOSP)"
grant_permissions "com.google.android.apps.photos" "Photos"
grant_permissions "com.android.camera2" "Camera"
grant_permissions "com.google.android.gm" "Gmail"
grant_permissions "com.google.android.calendar" "Google Calendar"

echo ""
echo "✓ Permissions granted!"
echo ""

# ===========================================
# PART 2: Chrome First-Run Setup
# ===========================================

echo "--- Setting up Chrome ---"

# Method 1: Use command-line flags to skip first-run
# This writes Chrome's first-run sentinel file
echo "Skipping Chrome first-run experience..."
$ADB shell "am start -n com.android.chrome/com.google.android.apps.chrome.Main --ez no_first_run true --ez disable_fre_and_default_browser_prompt true" 2>/dev/null || true
sleep 2

# Method 2: Accept Chrome ToS via shared preferences (if available)
# This marks the first-run as completed
$ADB shell "mkdir -p /data/data/com.android.chrome/shared_prefs" 2>/dev/null || true

# Method 3: Click through first-run dialog if it appears
# Launch Chrome and handle dialogs
echo "Launching Chrome to accept dialogs..."
$ADB shell am start -a android.intent.action.VIEW -d "about:blank" -p com.android.chrome 2>/dev/null || true
sleep 3

# Click "Accept & Continue" button (typical position for first ToS dialog)
# These coordinates work for common emulator resolutions
echo "  Attempting to dismiss Chrome dialogs..."

# Try clicking on common button positions for "Accept" or "Continue"
# Bottom center area where buttons typically appear
$ADB shell input tap 540 1800 2>/dev/null || true  # Accept button
sleep 1
$ADB shell input tap 540 1900 2>/dev/null || true  # Continue button
sleep 1
$ADB shell input tap 540 1700 2>/dev/null || true  # No thanks (sync)
sleep 1

# For higher resolution screens
$ADB shell input tap 540 2100 2>/dev/null || true
sleep 1

# Press back to exit any remaining dialogs
$ADB shell input keyevent KEYCODE_BACK 2>/dev/null || true
sleep 0.5
$ADB shell input keyevent KEYCODE_HOME 2>/dev/null || true

echo "  ✓ Chrome setup attempted"
echo ""

# ===========================================
# PART 3: Markor First-Run Tutorial
# ===========================================

echo "--- Setting up Markor ---"
$ADB shell am start -n net.gsantner.markor/.activity.MainActivity 2>/dev/null || true
sleep 2

# Dismiss tutorial screens (arrow button position varies by resolution)
echo "  Dismissing Markor tutorial..."
for i in {1..5}; do
    # Common positions for "Next" arrow button
    $ADB shell input tap 979 2250 2>/dev/null || true  # 1080p
    $ADB shell input tap 960 1700 2>/dev/null || true  # 720p
    sleep 1
done

$ADB shell input keyevent KEYCODE_HOME 2>/dev/null || true
echo "  ✓ Markor setup attempted"
echo ""

# ===========================================
# PART 4: Other App First-Run Dismissals
# ===========================================

echo "--- Setting up other apps ---"

# Simple Gallery Pro
echo "  Setting up Simple Gallery..."
$ADB shell am start -n com.simplemobiletools.gallery.pro/.activities.MainActivity 2>/dev/null || true
sleep 2
$ADB shell input tap 540 1900 2>/dev/null || true  # "Got it" button
sleep 1
$ADB shell input keyevent KEYCODE_HOME 2>/dev/null || true

# VLC
echo "  Setting up VLC..."
$ADB shell am start -n org.videolan.vlc/.StartActivity 2>/dev/null || true
sleep 2
$ADB shell input tap 540 1900 2>/dev/null || true  # Dismiss any welcome
sleep 1
$ADB shell input keyevent KEYCODE_BACK 2>/dev/null || true
$ADB shell input keyevent KEYCODE_HOME 2>/dev/null || true

# OsmAnd
echo "  Setting up OsmAnd..."
$ADB shell am start -n net.osmand.plus/net.osmand.plus.activities.MapActivity 2>/dev/null || true
sleep 3
$ADB shell input tap 540 1900 2>/dev/null || true  # Skip intro
sleep 1
$ADB shell input tap 540 1900 2>/dev/null || true
sleep 1
$ADB shell input keyevent KEYCODE_HOME 2>/dev/null || true

echo "  ✓ Other apps setup attempted"
echo ""

# ===========================================
# PART 5: Device Settings
# ===========================================

echo "--- Configuring device settings ---"

# Disable screen lock
$ADB shell settings put secure lock_screen_lock_after_timeout 2147483647
$ADB shell settings put system screen_off_timeout 2147483647

# Disable animations for faster testing
$ADB shell settings put global window_animation_scale 0
$ADB shell settings put global transition_animation_scale 0
$ADB shell settings put global animator_duration_scale 0

# Enable stay awake when charging
$ADB shell settings put global stay_on_while_plugged_in 7

# Disable auto-rotate (optional, helps with consistent screenshots)
$ADB shell settings put system accelerometer_rotation 0

echo "  ✓ Device settings configured"
echo ""

echo "========================================"
echo "✓ All setup complete!"
echo "========================================"
echo ""
echo "Notes:"
echo "  - Some apps may still show dialogs on first use"
echo "  - Button coordinates assume 1080x1920 or similar resolution"
echo "  - For different resolutions, adjust tap coordinates"
echo "  - Run this script after each emulator wipe/reset"
echo ""
