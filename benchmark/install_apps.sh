#!/bin/bash
set -e

ADB="${ADB:-adb}"
APK_DIR="${APK_DIR:-$HOME/tmp/android_world/app_data}"
cd "$APK_DIR"

echo "========================================"
echo "Installing Android World Apps"
echo "This will take approximately 15-20 minutes"
echo "========================================"
echo ""

install_app() {
    local num=$1
    local total=$2
    local name=$3
    local apk=$4
    local wait=$5
    
    echo "[$num/$total] Installing $name..."
    if $ADB install -r -t -g "$apk"; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
        return 1
    fi
    echo "  Waiting ${wait}s for system to stabilize..."
    sleep "$wait"
}

install_app 1 17 "AndroidWorld" "androidworld.apk" 10
install_app 2 17 "Clipper" "clipper.apk" 10
install_app 3 17 "Audio Recorder" "com.dimowner.audiorecorder_926.apk" 10
install_app 4 17 "Pro Expense" "com.arduia.expense_11.apk" 15
install_app 5 17 "Markor" "net.gsantner.markor_146.apk" 15
install_app 6 17 "Broccoli" "com.flauschcode.broccoli_1020600.apk" 15
install_app 7 17 "Retro Music" "code.name.monkey.retromusic_10603.apk" 15
install_app 8 17 "Simple Calendar" "com.simplemobiletools.calendar.pro_238.apk" 15
install_app 9 17 "Simple Draw" "com.simplemobiletools.draw.pro_79.apk" 15
install_app 10 17 "Simple SMS" "com.simplemobiletools.smsmessenger_85.apk" 15
install_app 11 17 "OpenTracks" "de.dennisguse.opentracks_5705.apk" 15
install_app 12 17 "MiniWoB" "miniwobapp.apk" 15
install_app 13 17 "Tasks" "org.tasks_130605.apk" 15
install_app 14 17 "Simple Gallery (31MB)" "com.simplemobiletools.gallery.pro_396.apk" 20
install_app 15 17 "VLC (37MB)" "org.videolan.vlc_13050408.apk" 20
install_app 16 17 "Joplin (42MB)" "net.cozic.joplin_2097740.apk" 25
install_app 17 17 "OsmAnd (336MB)" "net.osmand-4.6.13.apk" 30

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Verifying installed apps..."
installed_count=$($ADB shell pm list packages | grep -E "(markor|expense|vlc|retro|gallery|draw|calendar|sms|recorder|osmand|broccoli|joplin|tasks|miniwob|androidworld|clipper)" | wc -l)
echo "Installed packages: $installed_count / 17"
echo ""
echo "All apps successfully installed!"
echo "You can now run the benchmark without --perform_emulator_setup flag"
