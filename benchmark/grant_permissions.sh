#!/bin/bash

ADB="${ADB:-adb}"

echo "========================================"
echo "Granting Permissions for Android World Apps"
echo "========================================"
echo ""

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

# Grant permissions for all Android World apps
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

echo ""
echo "========================================"
echo "Permissions Granted!"
echo "========================================"
echo ""
echo "Setting additional device settings..."

# Disable screen lock
$ADB shell settings put secure lock_screen_lock_after_timeout 2147483647
$ADB shell settings put system screen_off_timeout 2147483647

# Disable animations for faster testing
$ADB shell settings put global window_animation_scale 0
$ADB shell settings put global transition_animation_scale 0
$ADB shell settings put global animator_duration_scale 0

# Enable stay awake when charging
$ADB shell settings put global stay_on_while_plugged_in 7

echo "✓ Device settings configured"
echo ""
echo "All setup complete! Ready to run benchmark."
